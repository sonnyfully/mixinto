"""CLI interface for mixinto."""
import json
import sys
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from mixinto.core import analyze_audio, extend_audio
from mixinto.io.audio.writer import write_audio
from mixinto.utils.types import ExtendRequest

app = typer.Typer(
    name="mixinto",
    add_completion=False,
    help="DJ-first tool that extends track intros into longer, mix-safe lead-ins.",
)

# Default directories relative to project root
DEFAULT_INPUT_DIR = Path("input")
DEFAULT_OUTPUT_DIR = Path("output")


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for pyproject.toml or setup.py
    # Start from the cli.py file location and walk up
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    
    # Also try from current working directory
    cwd = Path.cwd()
    current = cwd
    while current.parent != current:
        if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def resolve_input_path(file_path: str | None) -> Path:
    """
    Resolve input file path, checking default input directory if path is relative.
    
    Args:
        file_path: Input file path (can be None, relative, or absolute)
    
    Returns:
        Resolved Path object
    """
    if file_path is None:
        raise ValueError("Input file path is required")
    
    path = Path(file_path)
    
    # If absolute path, use as-is
    if path.is_absolute():
        return path
    
    # If relative path exists as-is, use it
    if path.exists():
        return path.resolve()
    
    # Otherwise, try in default input directory
    project_root = get_project_root()
    default_input = project_root / DEFAULT_INPUT_DIR / path
    if default_input.exists():
        return default_input.resolve()
    
    # Return the original path (will be validated later)
    return path.resolve() if path.exists() else path


def resolve_output_path(output_path: str | None, input_path: Path, suffix: str = "_extended") -> Path:
    """
    Resolve output file path, defaulting to output directory if not specified.
    
    Args:
        output_path: Output file path (can be None, relative, or absolute)
        input_path: Input file path (used to generate default output name)
        suffix: Suffix to add to input filename for default output name
    
    Returns:
        Resolved Path object
    """
    if output_path is None:
        # Generate default output path
        project_root = get_project_root()
        default_output_dir = project_root / DEFAULT_OUTPUT_DIR
        default_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename based on input filename
        input_stem = input_path.stem
        input_suffix = input_path.suffix or ".wav"
        output_filename = f"{input_stem}{suffix}{input_suffix}"
        return default_output_dir / output_filename
    
    path = Path(output_path)
    
    # If absolute path, use as-is
    if path.is_absolute():
        return path
    
    # If relative path, check if it's in output directory or use as-is
    project_root = get_project_root()
    default_output = project_root / DEFAULT_OUTPUT_DIR / path
    
    # If the path looks like just a filename, put it in output directory
    if not path.parent or path.parent == Path("."):
        return default_output.resolve()
    
    # Otherwise use the path as-is
    return path.resolve()


class ExitCode:
    """Exit codes for the CLI."""
    SUCCESS = 0  # Operation completed successfully
    REFUSED = 1  # Ran correctly but negative outcome (e.g., refusal to extend)
    USER_ERROR = 2  # User error (invalid input, file not found, etc.)
    INTERNAL_ERROR = 3  # Internal error (unexpected exception, etc.)


def write_json_report(data: dict[str, Any], output_path: str | None, pretty: bool = True) -> None:
    """Write JSON report to file or stdout."""
    json_str = json.dumps(data, indent=2 if pretty else None, ensure_ascii=False)
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json_str, encoding="utf-8")
        typer.echo(f"Report written to: {output_file}", err=True)
    else:
        typer.echo(json_str)


@app.command()
def analyze(
    file: str = typer.Argument(..., help="The input audio file to analyze. If relative, will check in 'input/' directory."),
    preset: str = typer.Option("dj_safe", "--preset", "-p", help="The preset name."),
    report: str | None = typer.Option(None, "--report", "-r", help="Output path for JSON report. If not specified, prints to stdout."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON (always enabled if --report is used)."),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty print JSON output."),
) -> None:
    """
    Analyze an audio file to detect BPM, downbeats, and intro characteristics.
    
    Returns analysis results including beat grid, intro profile, and mix safety metrics.
    """
    input_path = resolve_input_path(file)
    
    # Validate input file exists
    if not input_path.exists():
        typer.echo(f"Error: Input file not found: {input_path}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    if not input_path.is_file():
        typer.echo(f"Error: Path is not a file: {input_path}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    try:
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
        
        # Output results
        if json_output or report:
            write_json_report(analysis_result, report, pretty)
        else:
            # Human-readable output
            typer.echo(f"Analysis complete for: {input_path}")
            typer.echo(f"Preset: {preset}")
            typer.echo(f"BPM: {beat_grid.bpm:.1f} (confidence: {beat_grid.confidence:.2f})")
            typer.echo(f"Beats: {len(beat_grid.beats_s)}, Downbeats: {len(beat_grid.downbeats_s)}")
            typer.echo(f"Intro window: {intro_profile.start_s:.2f}s - {intro_profile.end_s:.2f}s")
            typer.echo(f"Mix safety score: {intro_profile.mix_safety_score:.2f}")
            if intro_profile.flags:
                typer.echo(f"Flags: {', '.join(intro_profile.flags)}")
        
        sys.exit(ExitCode.SUCCESS)
        
    except ValidationError as e:
        typer.echo(f"Validation error: {e}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    except Exception as e:
        typer.echo(f"Internal error during analysis: {e}", err=True)
        sys.exit(ExitCode.INTERNAL_ERROR)


@app.command()
def extend(
    file: str = typer.Argument(..., help="The input audio file to extend. If relative, will check in 'input/' directory."),
    output: str | None = typer.Option(None, "--out", "-o", help="Output path for the extended audio file. If not specified, defaults to 'output/{input_filename}_extended.wav'."),
    bars: int | None = typer.Option(None, "--bars", "-b", help="Number of bars to extend the intro by."),
    seconds: float | None = typer.Option(None, "--seconds", "-s", help="Number of seconds to extend the intro by."),
    preset: str = typer.Option("dj_safe", "--preset", "-p", help="The preset name."),
    report: str | None = typer.Option(None, "--report", "-r", help="Output path for JSON report."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON (always enabled if --report is used)."),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty print JSON output."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output file if it exists."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Perform a dry run without generating output."),
) -> None:
    """
    Extend a track's intro into a longer, mix-safe lead-in.
    
    Generates new intro bars that preserve the track's tempo, groove, timbre, and energy
    while adding time you can comfortably mix into.
    """
    input_path = resolve_input_path(file)
    output_path = resolve_output_path(output, input_path)
    
    # Validate input file exists
    if not input_path.exists():
        typer.echo(f"Error: Input file not found: {input_path}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    if not input_path.is_file():
        typer.echo(f"Error: Path is not a file: {input_path}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    # Validate that either bars or seconds is provided
    if bars is None and seconds is None:
        typer.echo("Error: Either --bars or --seconds must be provided.", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    if bars is not None and seconds is not None:
        typer.echo("Error: Cannot specify both --bars and --seconds.", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    # Check if output file exists and overwrite flag
    if output_path.exists() and not overwrite:
        typer.echo(f"Error: Output file already exists: {output_path}. Use --overwrite to replace it.", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    try:
        # Validate request using Pydantic model
        extend_request = ExtendRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            preset=preset,
            target_bars=bars,
            target_seconds=seconds,
            overwrite=overwrite,
            dry_run=dry_run,
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
            
            # Output refusal report
            if json_output or report:
                write_json_report(extension_result, report, pretty)
            else:
                typer.echo(f"Extension refused: {metrics.get('refusal_reason', 'Unknown reason')}", err=True)
            
            sys.exit(ExitCode.REFUSED)
        
        # Success case - generate output
        if not dry_run:
            # Write extended audio file
            write_audio(extended_buffer, output_path, format="wav")
            typer.echo(f"Extended audio written to: {output_path}")
        
        # Build success result
        extension_result = {
            "status": "success",
            "input_file": str(input_path),
            "output_file": str(output_path),
            "preset": preset,
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
        
        # Output success report
        if json_output or report:
            write_json_report(extension_result, report, pretty)
        else:
            typer.echo(f"Extension complete: {input_path} -> {output_path}")
            if bars:
                typer.echo(f"Added {bars} bars")
            if seconds:
                typer.echo(f"Added {seconds} seconds")
            typer.echo(f"Mix safety score: {metrics.get('mix_safety_score', 0.0):.2f}")
        
        # Confirm output path
        if not dry_run and output_path.exists():
            typer.echo(f"Output confirmed: {output_path.absolute()}", err=True)
        
        sys.exit(ExitCode.SUCCESS)
        
    except ValidationError as e:
        typer.echo(f"Validation error: {e}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    except Exception as e:
        typer.echo(f"Internal error during extension: {e}", err=True)
        sys.exit(ExitCode.INTERNAL_ERROR)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
