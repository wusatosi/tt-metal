# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import click
import json
import hashlib
from elasticsearch import Elasticsearch, NotFoundError
from framework.elastic_config import *
from framework.sweeps_logger import sweeps_logger as logger
from datetime import datetime
import requests
import sys


@click.group(help="CLI Tool for managing associations between vector IDs and GitHub issues in Elasticsearch.")
@click.option("--module-name", default=None, help="Name of the module to query.")
@click.option("--suite-name", default=None, help="Filter by suite name.")
@click.option("--vector-id", default=None, help="Filter by an individual Vector ID.")
@click.option("--elastic", default="corp", help="Elastic Connection String. Available presets: ['corp', 'cloud'].")
@click.pass_context
def cli(ctx, module_name, suite_name, vector_id, elastic):
    """
    This CLI tool helps in managing associations between vector IDs and GitHub issues within Elasticsearch.
    You can:
    - Associate vector IDs with GitHub issues.
    - Disassociate existing associations.
    - Display current associations.
    """
    ctx.ensure_object(dict)
    ctx.obj["module_name"] = module_name
    ctx.obj["suite_name"] = suite_name
    ctx.obj["vector_id"] = vector_id
    ctx.obj["elastic"] = get_elastic_url(elastic)


def filter_input_params(vector):
    keys_to_remove = [
        "sweep_name",
        "suite_name",
        "vector_id",
        "input_hash",
        "timestamp",
        "tag",
        "invalid_reason",
        "status",
        "validity",
    ]
    for key in keys_to_remove:
        if key in vector:
            vector.pop(key)
    logger.debug(json.dumps(vector, indent=2))
    return vector


def fetch_vector_details(es, module_name, vector_id):
    vector_index = VECTOR_INDEX_PREFIX + module_name
    try:
        response = es.get(index=vector_index, id=vector_id)
        if response and response.get("found"):
            return filter_input_params(response["_source"])
    except NotFoundError:
        logger.error(f"Vector ID {vector_id} not found in module {module_name}.")
        sys.exit(1)
    return None


def generate_hash(data):
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()


def validate_github_issue_link(issue_link):
    response = requests.head(issue_link)
    return response.status_code == 200


def save_issue_to_elasticsearch(es, index, hash_value, issue_numbers):
    existing_doc = es.options(ignore_status=[404]).get(index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value)

    if existing_doc and "_source" in existing_doc:
        existing_issues = set(existing_doc["_source"].get("issues", []))
        existing_issues.update(issue_numbers)
    else:
        existing_issues = set(issue_numbers)
    # Convert issue numbers to GitHub issue links
    github_issue_links = []
    for issue in issue_numbers:
        issue_link = f"https://github.com/tenstorrent/tt-metal/issues/{issue}"
        if validate_github_issue_link(issue_link):
            github_issue_links.append(issue_link)
        else:
            logger.warning(f"GitHub issue link {issue_link} does not work.")

    doc = {
        "hash": hash_value,
        "issues": list(existing_issues),
        "issue_links": github_issue_links,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    es.index(index=index, id=hash_value, body=doc)
    logger.info(f"Updated hash {hash_value} with issues {list(existing_issues)} in Elasticsearch.")


def process_vector_associations(es, vector_ids, module_name, issue_numbers):
    for vector_id in vector_ids:
        vector_details = fetch_vector_details(es, module_name, vector_id)
        if not vector_details:
            continue

        logger.debug(f"Vector Details for {vector_id}: {json.dumps(vector_details, indent=2)}")
        hash_value = generate_hash(vector_details)
        logger.debug(f"hash_value: {hash_value}")
        save_issue_to_elasticsearch(es, RESULT_INDEX_PREFIX + "hash_table_index", hash_value, issue_numbers)


def split_comma_separated(ctx, param, value):
    if value:
        return value.split(",")
    return []


@cli.command(help="Associate one or more vector IDs with GitHub issue numbers.")
@click.option(
    "--vector-ids", required=True, help="Comma-separated list of Vector IDs to associate (e.g., 'id1,id2,id3')."
)
@click.option("--module", required=True, help="Module name to which the vector IDs belong.")
@click.option("--issues", required=True, help="Comma-separated list of GitHub Issue Numbers (e.g., '101,102').")
@click.pass_context
def associate_issues_with_vectors(ctx, vector_ids, module, issues):
    """
    This command associates one or more vector IDs with GitHub issue numbers.
    The associations are stored in Elasticsearch using unique hashes of vector details.
    """
    es = Elasticsearch(ctx.obj["elastic"], basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    vector_ids = vector_ids.split(",")
    issues = issues.split(",")
    process_vector_associations(es, vector_ids, module, issues)
    es.close()


@cli.command(help="Remove associations between vector IDs and GitHub issues.")
@click.option("--vector-ids", required=True, help="Comma-separated list of Vector IDs (e.g., 'id1,id2').")
@click.option("--module", required=True, help="Module name of the vector IDs.")
@click.option("--issues", required=True, help="Comma-separated list of GitHub Issue Numbers to remove.")
@click.pass_context
def disassociate_issues_from_vectors(ctx, vector_ids, module, issues):
    """
    This command removes the association between vector IDs and specified GitHub issues.
    Only the provided issues will be removed, preserving other existing associations.
    """

    vector_ids = vector_ids.split(",")
    issues = issues.split(",")
    es = Elasticsearch(ctx.obj["elastic"], basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    for vector_id in vector_ids:
        vector_details = fetch_vector_details(es, module, vector_id)
        if not vector_details:
            continue

        hash_value = generate_hash(vector_details)
        existing_doc = es.options(ignore_status=[404]).get(
            index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value
        )
        if existing_doc and "_source" in existing_doc:
            existing_issues = set(existing_doc["_source"].get("issues", []))
            updated_issues = existing_issues.difference(issues)
            doc = {
                "hash": hash_value,
                "issues": list(updated_issues),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            es.index(index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value, body=doc)
            logger.info(
                f"Updated hash {hash_value} for Vector ID {vector_id} with issues {list(updated_issues)} in Elasticsearch."
            )
        else:
            logger.warning(f"No existing document found for hash {hash_value}.")
    es.close()


@cli.command(help="Display existing issue associations for specified vector IDs.")
@click.option(
    "--vector-ids", required=True, help="Comma-separated list of Vector IDs to check associations (e.g., 'id1,id2')."
)
@click.option("--module", required=True, help="Module name of the vector IDs.")
@click.pass_context
def show_issue_associations(ctx, vector_ids, module):
    """
    This command displays the existing associations between the provided vector IDs and GitHub issues.
    """

    vector_ids = vector_ids.split(",")
    es = Elasticsearch(ctx.obj["elastic"], basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    for vector_id in vector_ids:
        vector_details = fetch_vector_details(es, module, vector_id)
        if not vector_details:
            continue

        logger.info(f"Vector Details for {vector_id}: {json.dumps(vector_details, indent=2)}")
        hash_value = generate_hash(vector_details)
        logger.debug(f"Generated hash value: {hash_value}")
        existing_doc = es.options(ignore_status=[404]).get(
            index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value
        )
        if existing_doc and "_source" in existing_doc:
            issues = existing_doc["_source"].get("issues", [])
            logger.info(f"Vector ID {vector_id} is associated with issues: {issues}")
        else:
            logger.warning(f"No existing document found for hash {hash_value}.")
    es.close()


if __name__ == "__main__":
    cli(obj={})
