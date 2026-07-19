# Pytest

Pytest is a powerful testing framework for Python that simplifies writing and executing test cases. It supports features like test discovery, fixtures, parameterized testing, and detailed assertions. For more details, visit the official [Pytest Documentation](https://docs.pytest.org/en/latest/).


## Testing with multiple input sets

The `@pytest.mark.parametrize` decorator allows you to run a single test function with multiple sets of inputs.

### Example

```python
@pytest.mark.parametrize("arg1, arg2, expected", [
    (1, 2, 3),
    (2, 3, 5),
    (3, 5, 8),
])
def test_addition(arg1, arg2, expected):
    assert arg1 + arg2 == expected
```

### Explanation
- This is particularly useful for testing a function with various combinations of arguments

## Marking specific parameters

You can use `pytest.param` to mark specific parameter combinations with additional metadata, such as expected failures (`xfail`).

### Example

```python
@pytest.mark.parametrize("inputs", [
    pytest.param(
        ((1, 2, 3), (4, 5, 6)), marks=pytest.mark.xfail(reason="reason"))
])
```

### Explanation
- In this example, the first parameter combination is marked as `xfail` with a reason provided, indicating it is expected to fail.
- This is useful when only some parameter sets are failing or not working correctly.

## Skipping tests

Use the `@pytest.mark.skip` decorator to skip a test.

### Example

```python
@pytest.mark.skip(reason="Causes segmentation fault")
def test_future_feature():
    assert some_function() == "expected result"
```

### Explanation
- Skipping tests is particularly useful when a test is causing crashes (e.g., segmentation faults) or breaking the CI pipeline.

## Marking tests as expected to fail

The `@pytest.mark.xfail` decorator marks a test that is expected to fail.

### Example

```python
@pytest.mark.xfail(reason="Known bug in version 1.2.3")
def test_known_bug():
    assert buggy_function() == "expected"
```

### Explanation
- If the test passes unexpectedly, pytest will flag it as `XPASS`.
- If the test `XPASS`, it indicates an unexpected pass and will be reported as an error.
- This is helpful when we need a reminder that a particular test is passing, especially in cases where it previously failed and we want to review all related instances or areas that experienced issues.

## Avoid adding decorators inside tests

### Example
```python
@pytest.mark.parametrize("model_path", ["<path>/model_path1", "<path>/model_path2"])
def test_model(model_path):
    if model_path == "<path>/model_path1":
        pytest.xfail("reason")
```

### Explanation
- In this example, one of the models fails a test. Using an `if` statement to apply `xfail` is problematic because it will always mark the test as failing, even if it passes.
- Instead, use `pytest.param` to explicitly define expected outcomes as shown in the recommended approach above. This ensures more accurate and reliable test behavior.
