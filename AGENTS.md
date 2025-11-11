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
