# HTML Parser Testing - Quick Reference

## When You Inspect a New Case

### 1. Add the Test Case

```bash
python add_test_case.py <CELEX_NUMBER>
```

Example:

```bash
python add_test_case.py 62010CJ0454
```

The script will:

- Extract all paragraphs using ECJProcessor
- Show you samples for verification
- Ask for confirmation
- Add to `test_cases.json`

### 2. Verify the Test Passes

```bash
pytest test_html_parser.py::test_paragraph_extraction[<CELEX>] -v
```

Example:

```bash
pytest test_html_parser.py::test_paragraph_extraction[62010CJ0454] -v
```

### 3. Run All Tests

```bash
pytest test_html_parser.py -v
```

## What Gets Tested

For each case, the tests verify:

✅ Correct number of paragraphs extracted
✅ Paragraph numbers match expected (1, 2, 3, ...)
✅ **Each paragraph's content matches exactly**

This is a complete regression test - if the parser changes and extracts different content, the test will fail.

## Common Commands

```bash
# Add a new test case
python add_test_case.py 62015CJ0123

# Add without confirmation prompt
python add_test_case.py 62015CJ0123 --no-confirm

# Specify custom path
python add_test_case.py 62015CJ0123 --path summaries/62015CJ0123.html

# Run all tests
pytest test_html_parser.py -v

# Run test for specific case
pytest test_html_parser.py::test_paragraph_extraction[62010CJ0454] -v
```

## Current Test Cases

| CELEX       | Paragraphs | Status     |
| ----------- | ---------- | ---------- |
| 62010CJ0454 | 28         | ✅ Passing |

## Files

- **test_cases.json** - All test cases with expected paragraphs
- **test_html_parser.py** - The test suite
- **add_test_case.py** - Script to add new test cases
- **QUICK_REFERENCE.md** - This file
