# Repository Guidelines

## Project Structure & Module Organization
- `pynebula/`: main Python package (optimal control, simulators, robots). Library code lives under `pynebula/pynebula/`. Tests live in `pynebula/tests/` and `pynebula/pynebula/ocp/tests/`.
- `mujoco_viewer/`: standalone MuJoCo viewer with its own `AGENTS.md`. Follow that file for changes in this subproject.
- `robot_description/`: MuJoCo/URDF models and meshes for the Z1 arm. Treat this as data-only; avoid code or tooling here.

## Build, Test, and Development Commands
- Set up a dev environment for `pynebula`:
  ```bash
  cd pynebula
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  pip install -e .
  ```
- Run tests (from `pynebula/`):
  ```bash
  pytest
  # or a subset, e.g.:
  pytest -m unit
  ```
- For `mujoco_viewer`, see `mujoco_viewer/AGENTS.md` and `mujoco_viewer/README.md` for run instructions.

## Coding Style & Naming Conventions
- Python: follow PEP 8 with 4-space indentation and type hints for new/modified code.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, ALL_CAPS for constants.
- Keep public APIs under `pynebula/pynebula/` stable; prefer small, composable functions and explicit imports.
- If available, format and lint before committing:
  ```bash
  black pynebula
  ruff check pynebula
  ```

## Testing Guidelines
- Use `pytest` with configuration in `pynebula/pytest.ini`.
- Place new tests in `pynebula/tests/` (integration/unit) or alongside modules (e.g., `pynebula/pynebula/ocp/tests/`), named `test_*.py`.
- Follow `pytest.ini` patterns: test classes `Test*`, functions `test_*`. Use markers like `unit`, `integration`, `slow` as appropriate.
- New features and bug fixes should include at least one failing test before the fix and passing tests after.

## Commit & Pull Request Guidelines
- Use Conventional Commits where possible: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, etc. Example: `feat(ocp): add warm-start option`.
- Keep PRs focused and small. Include: a clear summary, rationale, and links to any related issues.
- Ensure tests pass locally (`pytest` in `pynebula/` and any relevant subproject tests) and update documentation/README files when behavior changes.

## Agent-Specific Instructions
- Prefer minimal, targeted edits that respect existing structure and style. Do not reformat unrelated files.
- Subdirectory `AGENTS.md` files (e.g., `mujoco_viewer/AGENTS.md`) take precedence for changes within their scope.
- When adding dependencies for `pynebula`, update both `requirements.txt` and `environment.yml` and justify the addition in the PR description.
