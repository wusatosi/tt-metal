import subprocess

TEST_NAME = "test_all_gather_ring_async_on_T3K"
TEST_PATH = "tests/ttnn/unit_tests/operations/ccl/test_new_all_gather.py"  # Replace with the actual filename


for i in range(1, 101):
    print(f"🔁 Running pytest iteration #{i}")
    result = subprocess.run(["python3", "-m", "pytest", "-svv", f"{TEST_PATH}::{TEST_NAME}"])

    if result.returncode != 0:
        print(f"❌ Test failed on iteration #{i}")
        input("⏸ Press Enter to pause...")
        import pdb

        pdb.set_trace()
