import pytest

from benchmark.evaluator import compute_patch_similarity


def test_identical_patches_score_one():
    patch = "--- a/auth.py\n+++ b/auth.py\n@@ -1,3 +1,3 @@\n-def login():\n+def login(user):\n     pass"
    assert compute_patch_similarity(patch, patch) == 1.0


def test_completely_different_patches():
    generated = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    truth = "--- a/db.py\n+++ b/db.py\n@@ -1 +1 @@\n-def foo():\n+def bar():"
    score = compute_patch_similarity(generated, truth)
    assert score < 0.5


def test_empty_generated_patch():
    truth = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    assert compute_patch_similarity("", truth) == 0.0


def test_empty_ground_truth():
    generated = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    assert compute_patch_similarity(generated, "") == 0.0


def test_both_empty():
    assert compute_patch_similarity("", "") == 0.0


def test_error_string_patch():
    truth = "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"
    assert compute_patch_similarity("ERROR: connection timeout", truth) == 0.0


def test_partial_overlap():
    generated = (
        "--- a/auth.py\n+++ b/auth.py\n@@ -1,3 +1,3 @@\n"
        " def login(user, password):\n-    return False\n+    return check(user, password)\n"
    )
    truth = (
        "--- a/auth.py\n+++ b/auth.py\n@@ -1,3 +1,4 @@\n"
        " def login(user, password):\n-    return False\n+    result = check(user, password)\n+    return result\n"
    )
    score = compute_patch_similarity(generated, truth)
    assert 0.0 < score < 1.0
