# Circuit-Tracer AI Copilot Instructions

## 🧠 Project Context & Architecture

**Context**: The current Circuit-Tracer library trace circuit in language models using (cross-layer) MLP transcoders. My goal is to extend it to use other dictionary learning methods different from Transcoders. 
**Core Workflow**:
1.  **Attribution**: Compute effects between transcoder features/errors/tokens (`circuit_tracer.attribution`).
2.  **Graph Generation**: Prune and format attribution data into a graph structure (`circuit_tracer.graph`).
3.  **Visualization**: Serve graph data via a local web server (`circuit_tracer.frontend`).

**Key Components**:
-   `circuit_tracer`: Main Python package.
-   `transcoder`: Handles `CrossLayerTranscoder`, activation functions, and model wrapping.
-   `frontend`: Web visualization assets (HTML/JS) and `local_server.py`.
-   `demos`: Juypter notebooks for tutorials and experiments.

**Data Flow**:
`Model + Prompts` -> `Attribution` -> `Raw Graph (.pt)` -> `Graph Pruning` -> `Vis-ready JSON` -> `Web UI`

## 🛠️ Development Workflow

### Environment & Dependencies
-   **Package Manager**: Standard `pip`. Install with dev deps: `pip install -e ".[dev]"`.
-   **Virtual Env**: User prefers **uvenv**.
-   **Compute**: Code often runs on SLURM clusters (DGX/H100).
    -   Group: "LADE"
    -   Submit: `sbatch slurm_scripts/slurm_script.sh` (if available) or interactively.
-   **Core Libs**: `transformer-lens` (model), `torch`, `einops`, `huggingface_hub`.
-   **Future/Preferred Libs**: `saelens` (for SAEs), `rich` (formatting), `plotly` (plotting).

### Verification
-   **Linting**: Run `ruff check .` and `ruff format .` before committing.
    -   *Strict rules*: Use modern union types (`|`), avoiding `typing.Union/List/Dict/Optional`.
-   **Testing**: Run `pytest`.
    -   Add unit tests in `tests/` for new `src/` functionality.
    -   Ensure demo notebooks (`demos/*.ipynb`) execute correctly after changes.

## 📝 Coding Conventions

### Style & Types
-   **Python Version**: `>=3.10`.
-   **Type Hints**: REQUIRED for all function signatures and class attributes.
    -   Use `X | Y` instead of `Union[X, Y]`.
    -   Use `list[int]` instead of `List[int]`.
    -   Use `dict[str, Any]` instead of `Dict[str, Any]`.
    -   Use `X | None` instead of `Optional[X]`.
-   **Documentation**: Comprehensive docstrings for all classes and functions.
-   **Structure**:
    -   Prefer modular components with clear separation of concerns (KISS).
    -   Separate visualization logic (`frontend`) from core math (`attribution`/`transcoder`).

### Project-Specific Patterns
-   **Transcoders**: Use `circuit_tracer.transcoder` classes. Avoid raw PyTorch hooks unless necessary; use `TransformerLens` hook points.
-   **Graph Data**: Visualization data is serialized to JSON. If modifying the graph structure, ensure `frontend/assets/*.js` handlers are compatible.
-   **CLI**: The entry point is `circuit_tracer.__main__`. New commands should be added here.

## ⚠️ Critical Files & Directories
-   `pyproject.toml`: Dependency and tool configuration (Ruff settings).
-   `circuit_tracer/frontend/local_server.py`: Visualization server logic.
-   `circuit_tracer/attribution/attribute.py`: Core attribution algorithm.
-   `demos/`: Verification notebooks (must stay functional).
