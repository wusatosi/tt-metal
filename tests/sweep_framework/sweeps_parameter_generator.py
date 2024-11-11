# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import importlib
import pathlib
import datetime
import os
import hashlib
import json

from framework.permutations import *
from framework.serialize import serialize
from framework.statuses import VectorValidity, VectorStatus
from framework.sweeps_logger import sweeps_logger as logger

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


# Generate vectors from module parameters
def generate_vectors(module_name, start_x=1, end_x=9, start_y=1):
    test_module = importlib.import_module("sweeps." + module_name)
    parameters = test_module.get_parameters(start_x, end_x, start_y)

    for suite in parameters:
        logger.info(f"Generating test vectors for suite {suite}.")
        suite_vectors = list(permutations(parameters[suite]))
        for v in suite_vectors:
            v["suite_name"] = suite
            v["validity"] = VectorValidity.VALID
            v["invalid_reason"] = ""
            v["status"] = VectorStatus.CURRENT
            v["sweep_name"] = module_name

        suite_vectors = invalidate_vectors(test_module, suite_vectors)
        suite_vectors = downsample(suite, suite_vectors)
        if DUMP_FILE:
            export_suite_vectors_json(module_name, suite, suite_vectors)
        else:
            export_suite_vectors(module_name, suite, suite_vectors)


def downsample(suite, suite_vectors, min_desired=3000):
    starting_vecs = len(suite_vectors)
    while len(suite_vectors) > 2 * min_desired:
        del suite_vectors[::2]
    if starting_vecs != len(suite_vectors):
        logger.info(f"Downsampled suite {suite} vectors from {starting_vecs} to {len(suite_vectors)}")
    return suite_vectors


# Perform any post-gen validation to the resulting vectors.
def invalidate_vectors(test_module, vectors, delete_invalid_vec=True):
    if "invalidate_vector" not in dir(test_module):
        return
    for vector in vectors:
        invalid, reason = test_module.invalidate_vector(vector)
        if invalid:
            vector["validity"] = VectorValidity.INVALID
            vector["invalid_reason"] = reason

    if delete_invalid_vec:
        logger.info(f"SWEEPS: Skipping writing out invalid test vectors.")
        starting_vecs = len(vectors)
        vectors = [v for v in vectors if v["validity"] == VectorValidity.VALID]
        logger.info(f"SWEEPS: {len(vectors)} of {starting_vecs} generated vectors were valid")

    return vectors


def export_suite_vectors_json(module_name, suite_name, vectors):
    input_grid = suite_name.split("-")[0]
    output_grid = suite_name.split("-")[1]
    EXPORT_DIR_PATH = SWEEPS_DIR / "vectors_export" / input_grid
    EXPORT_PATH = EXPORT_DIR_PATH / str(f"reshard-{input_grid}-{output_grid}.json")
    if not EXPORT_DIR_PATH.exists():
        os.makedirs(EXPORT_DIR_PATH)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    serialized_vectors = dict()
    warnings = []
    for i in range(len(vectors)):
        vector = dict()
        for elem in vectors[i].keys():
            vector[elem] = serialize(vectors[i][elem], warnings)
        input_hash = hashlib.sha224(str(vector).encode("utf-8")).hexdigest()
        vector["timestamp"] = current_time
        vector["input_hash"] = input_hash
        vector["tag"] = SWEEPS_TAG
        serialized_vectors[input_hash] = vector

    if EXPORT_PATH.exists():
        with open(EXPORT_PATH, "r") as file:
            data = json.load(file)
        with open(EXPORT_PATH, "w") as file:
            data[suite_name] = serialized_vectors
            json.dump(data, file, indent=2)
    else:
        with open(EXPORT_PATH, "w") as file:
            json.dump({suite_name: serialized_vectors}, file, indent=2)
    logger.info(f"SWEEPS: Generated {len(vectors)} test vectors for suite {suite_name}.")


# Output the individual test vectors.
def export_suite_vectors(module_name, suite_name, vectors):
    # Perhaps we export with some sort of readable id, which can be passed to a runner to run specific sets of input vectors. (export seed as well for reproducability)
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

    index_name = VECTOR_INDEX_PREFIX + module_name
    warnings = []

    try:
        response = client.search(
            index=index_name,
            query={
                "bool": {
                    "must": [
                        {"match": {"tag.keyword": SWEEPS_TAG}},
                        {"match": {"status.keyword": str(VectorStatus.CURRENT)}},
                        {"match": {"suite_name.keyword": suite_name}},
                    ]
                }
            },
            size=10000,
        )["hits"]["hits"]
        old_vector_ids = set(vector["_id"] for vector in response)
        old_vector_hashes = set(vector["_source"]["input_hash"] for vector in response)
    except NotFoundError as e:
        old_vector_ids = set()
        old_vector_hashes = set()
        pass

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    new_vector_hashes = set()
    serialized_vectors = dict()
    for i in range(len(vectors)):
        vector = dict()
        for elem in vectors[i].keys():
            vector[elem] = serialize(vectors[i][elem], warnings)
        input_hash = hashlib.sha224(str(vector).encode("utf-8")).hexdigest()
        new_vector_hashes.add(input_hash)
        vector["timestamp"] = current_time
        vector["input_hash"] = input_hash
        vector["tag"] = SWEEPS_TAG
        serialized_vectors[input_hash] = vector

    if old_vector_hashes == new_vector_hashes:
        logger.info(
            f"Vectors generated for module {module_name}, suite {suite_name} already exist with tag {SWEEPS_TAG}, and have not changed. ({len(old_vector_hashes)} existing tests). Skipping..."
        )
        return
    else:
        logger.info(
            f"New vectors found for module {module_name}, suite {suite_name}, with tag {SWEEPS_TAG}. Archiving old vectors and saving new suite. This step may take several minutes."
        )
        for old_vector_id in old_vector_ids:
            client.update(index=index_name, id=old_vector_id, doc={"status": str(VectorStatus.ARCHIVED)})
        serialized_vectors = list(serialized_vectors.values())
        while serialized_vectors != []:
            bulk = serialized_vectors[: min(200, len(serialized_vectors))]
            serialized_vectors = serialized_vectors[min(200, len(serialized_vectors)) :]
            bulk_query = []
            for vector in bulk:
                bulk_query.append({"create": {"_index": index_name}})
                bulk_query.append(vector)
            client.bulk(index=index_name, body=bulk_query)

        logger.info(f"SWEEPS: Generated {len(new_vector_hashes)} test vectors for suite {suite_name}.")


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name, start_x, end_x, start_y):
    if not module_name:
        for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
            module_name = str(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR))[:-3].replace("/", ".")
            logger.info(f"Generating test vectors for module {module_name}.")
            generate_vectors(module_name, start_x, end_x, start_y)
            logger.info(f"Finished generating test vectors for module {module_name}.\n\n")
    else:
        logger.info(f"Generating test vectors for module {module_name}.")
        generate_vectors(module_name, start_x, end_x, start_y)


def clean_module(module_name):
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    vector_index = VECTOR_INDEX_PREFIX + module_name

    if not client.indices.exists(index=vector_index):
        logger.error(f"Could not clean vectors for module {module_name} as there is no corresponding index.")
        exit(1)

    update_script = {"source": f"ctx._source.status = '{str(VectorStatus.ARCHIVED)}'", "lang": "painless"}
    client.update_by_query(
        index=vector_index, query={"match": {"tag.keyword": SWEEPS_TAG}}, script=update_script, refresh=True
    )
    logger.info(
        f"Marked all vectors with tag {SWEEPS_TAG} in index {vector_index} as archived. Proceeding with generation..."
    )

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Vector Generator",
        description="Generate test vector suites for the specified module.",
    )

    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        required=False,
        help="Must be set with module_name. Setting this flag will mark ALL old vectors for an sweep as Archived, and generate the new set. Use this if you make a mistake when generating your vectors and want to refresh to the current state of your op test file.",
    )
    parser.add_argument(
        "--elastic",
        required=False,
        default="corp",
        help="Elastic Connection String for vector database. Available presets are ['corp', 'cloud']",
    )
    parser.add_argument(
        "--tag",
        required=False,
        default=os.getenv("USER"),
        help="Custom tag for the vectors you are generating. This is to keep copies seperate from other people's test vectors. By default, this will be your username. You are able to specify a tag when running tests using the runner.",
    )
    parser.add_argument("--explicit", required=False, action="store_true")
    parser.add_argument(
        "--dump-file",
        required=False,
        action="store_true",
        help="If set, this will not use the ES database, and will instead dump tests to JSON.",
    )
    parser.add_argument("-x", type=int)
    parser.add_argument("-y", type=int)

    args = parser.parse_args(sys.argv[1:])

    global DUMP_FILE
    if not args.dump_file:
        from elasticsearch import Elasticsearch, NotFoundError
        from framework.elastic_config import *

        global ELASTIC_CONNECTION_STRING
        ELASTIC_CONNECTION_STRING = get_elastic_url(args.elastic)
        DUMP_FILE = False
    else:
        DUMP_FILE = True

    global SWEEPS_TAG
    SWEEPS_TAG = args.tag

    if args.tag == "ci-main" and not args.explicit:
        logger.error("The ci-main tag is reserved for CI only.")
        exit(1)

    logger.info(f"Running current generation with tag: {SWEEPS_TAG}.")

    if args.clean and not args.module_name:
        logger.error("The clean flag must be set in conjunction with a module name.")
        exit(1)
    elif args.clean:
        clean_module(args.module_name)

    generate_tests(args.module_name, args.x, args.x + 1, args.y)
