"""Flask web application for mixinto."""
import json
import random
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError

from mixinto.core import analyze_audio, extend_audio
from mixinto.io.audio.writer import write_audio
from mixinto.utils.types import ExtendRequest

app = Flask(__name__, static_folder=None)
CORS(app)  # Enable CORS for development

# Default directories relative to project root
DEFAULT_INPUT_DIR = Path("input")
DEFAULT_OUTPUT_DIR = Path("output")


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    
    cwd = Path.cwd()
    current = cwd
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    
    return Path.cwd()


def resolve_input_path(file_path: str | None) -> Path:
    """Resolve input file path, checking default input directory if path is relative."""
    if file_path is None:
        raise ValueError("Input file path is required")
    
    path = Path(file_path)
    
    if path.is_absolute():
        return path
    
    if path.exists():
        return path.resolve()
    
    project_root = get_project_root()
    default_input = project_root / DEFAULT_INPUT_DIR / path
    if default_input.exists():
        return default_input.resolve()
    
    return path.resolve() if path.exists() else path


def resolve_output_path(output_path: str | None, input_path: Path, suffix: str = "_extended") -> Path:
    """Resolve output file path, defaulting to output directory if not specified."""
    if output_path is None:
        project_root = get_project_root()
        default_output_dir = project_root / DEFAULT_OUTPUT_DIR
        default_output_dir.mkdir(parents=True, exist_ok=True)
        
        input_stem = input_path.stem
        input_suffix = input_path.suffix or ".wav"
        output_filename = f"{input_stem}{suffix}{input_suffix}"
        return default_output_dir / output_filename
    
    path = Path(output_path)
    
    if path.is_absolute():
        return path
    
    project_root = get_project_root()
    default_output = project_root / DEFAULT_OUTPUT_DIR / path
    
    if not path.parent or path.parent == Path("."):
        return default_output.resolve()
    
    return path.resolve()


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    """Serve the React app and static files."""
    frontend_dist = get_project_root() / "frontend" / "dist"
    
    if not frontend_dist.exists():
        return jsonify({
            "error": "Frontend not built. Run 'npm run build' in the frontend directory.",
            "hint": "For development, run 'npm run dev' in the frontend directory and access it directly."
        }), 404
    
    # Serve index.html for all routes (React Router support)
    if path == "" or not (frontend_dist / path).exists():
        return send_from_directory(str(frontend_dist), "index.html")
    
    # Serve static assets
    return send_from_directory(str(frontend_dist), path)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for analyzing audio."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        file = data.get("file")
        if not file:
            return jsonify({"error": "File parameter is required"}), 400
        
        preset = data.get("preset", "dj_safe")
        report = data.get("report")
        json_output = data.get("json_output", False) or report is not None
        pretty = data.get("pretty", True)
        
        input_path = resolve_input_path(file)
        
        if not input_path.exists():
            return jsonify({"error": f"Input file not found: {input_path}"}), 404
        
        if not input_path.is_file():
            return jsonify({"error": f"Path is not a file: {input_path}"}), 400
        
        # Perform analysis
        buffer, beat_grid, intro_profile = analyze_audio(
            input_path,
            preset_name=preset,
        )
        
        # Build analysis result
        analysis_result = {
            "status": "success",
            "input_file": str(input_path),
            "preset": preset,
            "audio_metadata": {
                "source_path": buffer.meta.source_path,
                "sample_rate": buffer.meta.sample_rate,
                "channels": buffer.meta.channels,
                "duration_s": buffer.meta.duration_s,
                "format": buffer.meta.format,
            },
            "beat_grid": {
                "bpm": beat_grid.bpm,
                "confidence": beat_grid.confidence,
                "beats_count": len(beat_grid.beats_s),
                "downbeats_count": len(beat_grid.downbeats_s),
            },
            "intro_profile": {
                "start_s": intro_profile.start_s,
                "end_s": intro_profile.end_s,
                "energy_mean": intro_profile.energy_mean,
                "energy_std": intro_profile.energy_std,
                "spectral_stability": intro_profile.spectral_stability,
                "rhythm_stability": intro_profile.rhythm_stability,
                "vocal_presence": intro_profile.vocal_presence,
                "mix_safety_score": intro_profile.mix_safety_score,
                "flags": intro_profile.flags,
            },
            "warnings": intro_profile.flags.copy(),
            "errors": [],
        }
        
        # Write report if requested
        if report:
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(analysis_result, indent=2 if pretty else None, ensure_ascii=False),
                encoding="utf-8"
            )
            analysis_result["report_path"] = str(report_path)
        
        return jsonify(analysis_result)
        
    except ValidationError as e:
        return jsonify({"error": f"Validation error: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error during analysis: {e}"}), 500


@app.route("/api/extend", methods=["POST"])
def api_extend():
    """API endpoint for extending audio."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        file = data.get("file")
        if not file:
            return jsonify({"error": "File parameter is required"}), 400
        
        output = data.get("output")
        bars = data.get("bars")
        seconds = data.get("seconds")
        preset = data.get("preset", "dj_safe")
        backend = data.get("backend", "baseline")
        seed = data.get("seed", "0")
        report = data.get("report")
        json_output = data.get("json_output", False) or report is not None
        pretty = data.get("pretty", True)
        overwrite = data.get("overwrite", False)
        dry_run = data.get("dry_run", False)
        
        input_path = resolve_input_path(file)
        output_path = resolve_output_path(output, input_path)
        
        if not input_path.exists():
            return jsonify({"error": f"Input file not found: {input_path}"}), 404
        
        if not input_path.is_file():
            return jsonify({"error": f"Path is not a file: {input_path}"}), 400
        
        if bars is None and seconds is None:
            return jsonify({"error": "Either bars or seconds must be provided"}), 400
        
        if bars is not None and seconds is not None:
            return jsonify({"error": "Cannot specify both bars and seconds"}), 400
        
        if output_path.exists() and not overwrite:
            return jsonify({"error": f"Output file already exists: {output_path}. Use overwrite to replace it."}), 400
        
        # Handle seed
        if seed.lower() == "random":
            actual_seed = random.randint(0, 2**31 - 1)
        else:
            try:
                actual_seed = int(seed)
            except ValueError:
                return jsonify({"error": f"Invalid seed value '{seed}'. Must be an integer or 'random'."}), 400
        
        # Validate request using Pydantic model
        extend_request = ExtendRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            preset=preset,
            target_bars=bars,
            target_seconds=seconds,
            overwrite=overwrite,
            dry_run=dry_run,
            backend=backend,
            seed=actual_seed,
        )
        
        # Perform extension
        extended_buffer, metrics = extend_audio(extend_request)
        
        # Check if extension was refused
        if metrics.get("refused", False):
            extension_result = {
                "status": "refused",
                "input_file": str(input_path),
                "output_file": str(output_path),
                "preset": preset,
                "target_bars": bars,
                "target_seconds": seconds,
                "extended": False,
                "refused": True,
                "refusal_reason": metrics.get("refusal_reason", "Track not safe to extend"),
                "warnings": metrics.get("warnings", []),
                "errors": metrics.get("errors", []),
                "metrics": {
                    "original_duration_s": metrics.get("original_duration_s", 0.0),
                    "extended_duration_s": metrics.get("extended_duration_s", 0.0),
                    "bars_added": 0,
                    "mix_safety_score": metrics.get("mix_safety_score", 0.0),
                    "seam_quality": metrics.get("seam_quality", 0.0),
                },
            }
            
            if report:
                report_path = Path(report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(extension_result, indent=2 if pretty else None, ensure_ascii=False),
                    encoding="utf-8"
                )
                extension_result["report_path"] = str(report_path)
            
            return jsonify(extension_result), 200
        
        # Success case - generate output
        if not dry_run:
            write_audio(extended_buffer, output_path, format="wav")
        
        # Build success result
        extension_result = {
            "status": "success",
            "input_file": str(input_path),
            "output_file": str(output_path),
            "preset": preset,
            "backend": backend,
            "seed": actual_seed,
            "seed_input": seed,
            "target_bars": bars,
            "target_seconds": seconds,
            "extended": True,
            "refused": False,
            "refusal_reason": None,
            "warnings": metrics.get("warnings", []),
            "errors": metrics.get("errors", []),
            "metrics": {
                "original_duration_s": metrics.get("original_duration_s", 0.0),
                "extended_duration_s": metrics.get("extended_duration_s", 0.0),
                "bars_added": metrics.get("bars_added", 0),
                "mix_safety_score": metrics.get("mix_safety_score", 0.0),
                "seam_quality": metrics.get("seam_quality", 0.0),
            },
        }
        
        if "bassline" in metrics:
            extension_result["bassline"] = metrics["bassline"]
        
        if report:
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(extension_result, indent=2 if pretty else None, ensure_ascii=False),
                encoding="utf-8"
            )
            extension_result["report_path"] = str(report_path)
        
        return jsonify(extension_result)
        
    except ValidationError as e:
        return jsonify({"error": f"Validation error: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal error during extension: {e}"}), 500


def main():
    """Run the Flask application."""
    import socket
    
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    
    print(f"Starting mixinto web interface on http://localhost:{port}")
    print(f"Press Ctrl+C to stop")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
