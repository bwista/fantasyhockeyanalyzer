# Testing Guide

This directory contains unit tests for the Fantasy Hockey Analyzer project.

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Tests with Coverage Report
```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_calculate_scoring_system.py
```

### Run Specific Test Class
```bash
pytest tests/test_calculate_scoring_system.py::TestCleanStatName
```

### Run Specific Test Method
```bash
pytest tests/test_calculate_scoring_system.py::TestCleanStatName::test_mapped_stat_names
```

## Test Structure

- `test_calculate_scoring_system.py`: Tests for the scoring system calculation logic
- `test_parse_draft_results.py`: Tests for draft results parsing functionality

## Coverage Reports

After running tests with coverage, an HTML report will be generated in `htmlcov/index.html` which you can open in a browser to see detailed coverage information.

## Adding New Tests

When adding new functionality to the project, please add corresponding tests:

1. Create a new test file with the naming convention `test_<module_name>.py`
2. Import the functions/classes you want to test
3. Create test classes with descriptive names
4. Write test methods that cover:
   - Normal operation cases
   - Edge cases
   - Error conditions
   - Input validation

## Best Practices

- Use descriptive test names that explain what is being tested
- Test both success and failure scenarios
- Use mocks for external dependencies (API calls, file I/O)
- Keep tests isolated and independent
- Use pytest fixtures for reusable test data 