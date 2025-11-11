# DeepXROMM Development Guidelines

## Build/Test Commands
- **Install dependencies**: `uv sync --locked --all-extras --dev`
- **Run all tests**: `uv run pytest --cov --cov-branch --cov-report=xml`
- **Run single test**: `uv run pytest tests/test_deepxromm.py::TestClassName::test_method` (for debugging only)
- **Format/lint**: `uv run pre-commit run -a` (uses Black)
- **Build package**: `python3 -m build`

## Code Style Guidelines
- **Formatting**: Black with Python 3.10 language version
- **Imports**: Standard library → third-party → local imports (each section sorted)
- **Type hints**: Use where beneficial, especially for public APIs and complex data structures
- **File paths**: Always use `pathlib.Path` instead of string paths
- **Logging**: Module-level logger with `logging.getLogger(__name__)`
- **Classes**: Use factory methods (`@classmethod`) for complex initialization
- **Error handling**: Specific exceptions with descriptive messages
- **Dependencies**: Use ruamel.yaml for YAML, pandas for data, cv2 for image processing
- **Naming**: snake_case for functions/variables, PascalCase for classes

## Version Control with Git

- **Branch Strategy**: 
  - `main` - Production-ready code (trunk)
  - Short-lived feature branches with descriptive names (e.g., `add_dark_mode`, `fix_memory_leak`)
  - Branch names should be concise, descriptive, and use underscores for spaces
  - No categorization prefixes in branch names (type is indicated in commit messages)
  - Branches should typically last no more than 1-2 days before merging

- **Commit Guidelines**:
  - Write meaningful commit messages with format: `<type>: <description>`
  - Types: feat, fix, docs, style, refactor, test, chore
  - Examples:
    - `feat: add xma_to_dlc frame selection algorithm`
    - `fix: correct index handling in data extraction`
    - `refactor: reduce nesting in xma_to_dlc function`
  - Keep commits focused on single logical changes
  - Reference issues in commit messages with #issue-number
  - Prefer smaller, frequent commits over large, infrequent ones

- **Pull Request Process**:
  - Create PR with clear title and description
  - Link related issues
  - Ensure CI tests pass
  - Request review from at least one team member
  - Merge frequently to minimize integration challenges

- **Code Review**:
  - Review for functionality, style, and test coverage
  - Address all comments before merging
  - Use "Request changes" for significant issues, "Comment" for minor suggestions

## Test Best Practices

- **Complete Testing Before Committing**:
  - **ALWAYS run the full test suite** before committing changes
  - **NEVER commit code that breaks existing tests**
  - Run both tests and quality checks with each change
  - Individual test execution is for debugging only, not verification

- **Acceptance Criteria**:
  - Define acceptance criteria before implementation
  - Use "Given-When-Then" format for scenarios
  - Document criteria in issue/PR description
  - Example:
    ```
    Given a data folder with multiple XMA trials
    When I call xma_to_dlc with nframes=100
    Then 100 frames are extracted evenly across trials
    ```

- **Test-First Development**:
  - Write failing test before implementation
  - Implement minimum code to pass test
  - Refactor while maintaining passing tests
  - Commit the test with its implementation

- **Test Coverage**:
  - Aim for >90% test coverage on new code
  - Include both unit tests and integration tests
  - Test edge cases and error conditions
  - Add tests for bug fixes to prevent regressions

- **Test Structure**:
  - Name tests clearly: `test_<function_name>_<scenario>`
  - Follow Arrange-Act-Assert pattern
  - Keep tests independent and idempotent
  - Use appropriate fixtures for test setup
  - Mock external dependencies when necessary

## Roles and Responsibilities

- **Repository Maintainers**:
  - Create and manage branches in the repository
  - Review and approve pull requests
  - Establish project roadmap and priorities
  - Ensure overall code quality and architecture

- **Build Agents and Contributors**:
  - Implement requested features and fixes
  - Run tests and quality checks before committing
  - Add and commit changes with appropriate messages
  - Do not push changes directly or create branches without approval
  - Document changes thoroughly in commit messages

## Pre-Commit Verification

Before committing any changes to the repository, build agents and contributors must:

1. **Test Suite Execution**:
   - Run and report the results of the full test suite
   - Provide a summary of test coverage metrics
   - Ensure all tests pass with no regressions

2. **Quality Check Execution**:
   - Run linting and formatting checks
   - Fix any style issues before committing
   - Report the results of quality checks

3. **Change Summary**:
   - Provide a concise summary of all changes made
   - Explain the purpose and impact of each change
   - List any new files created or existing files modified

4. **Commit Process**:
   - Only after verification steps 1-3 pass, proceed with:
     - Add the changed files using `git add`
     - Commit with descriptive message following the project format
     - DO NOT push changes (maintainers will handle this)

Example verification output:
```
Test Suite: PASSED (142 tests, 0 failures)
Coverage: 93% (increased by 2%)
Quality Checks: PASSED
Changes:
- Refactored xma_to_dlc() to reduce nesting
- Extracted 5 helper functions for better readability
- Added type hints to all public functions
Files modified:
- deepxromm/xrommtools.py
```
