"""CLI interface for mixinto."""
import json
import sys
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from mixinto.utils.types import ExtendRequest

app = typer.Typer(
    name="mixinto",
    add_completion=False,
    help="DJ-first tool that extends track intros into longer, mix-safe lead-ins.",
)


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
    file: str = typer.Argument(..., help="The input audio file to analyze."),
    preset: str = typer.Option("dj_safe", "--preset", "-p", help="The preset name."),
    report: str | None = typer.Option(None, "--report", "-r", help="Output path for JSON report. If not specified, prints to stdout."),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON (always enabled if --report is used)."),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty print JSON output."),
) -> None:
    """
    Analyze an audio file to detect BPM, downbeats, and intro characteristics.
    
    Returns analysis results including beat grid, intro profile, and mix safety metrics.
    """
    input_path = Path(file)
    
    # Validate input file exists
    if not input_path.exists():
        typer.echo(f"Error: Input file not found: {input_path}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    if not input_path.is_file():
        typer.echo(f"Error: Path is not a file: {input_path}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    
    try:
        # TODO: Implement actual analysis logic
        # This is a placeholder structure
        analysis_result = {
            "status": "success",
            "input_file": str(input_path),
            "preset": preset,
            "audio_metadata": {
                "source_path": str(input_path),
                "sample_rate": 44100,  # Placeholder
                "channels": 2,  # Placeholder
                "duration_s": 0.0,  # Placeholder
                "format": "wav",  # Placeholder
            },
            "beat_grid": {
                "bpm": 0.0,  # Placeholder
                "confidence": 0.0,  # Placeholder
                "beats_count": 0,  # Placeholder
                "downbeats_count": 0,  # Placeholder
            },
            "intro_profile": {
                "start_s": 0.0,  # Placeholder
                "end_s": 0.0,  # Placeholder
                "energy_mean": 0.0,  # Placeholder
                "energy_std": 0.0,  # Placeholder
                "spectral_stability": 0.0,  # Placeholder
                "rhythm_stability": 0.0,  # Placeholder
                "vocal_presence": 0.0,  # Placeholder
                "mix_safety_score": 0.0,  # Placeholder
                "flags": [],
            },
            "warnings": [],
            "errors": [],
        }
        
        # Output results
        if json_output or report:
            write_json_report(analysis_result, report, pretty)
        else:
            # Human-readable output (placeholder)
            typer.echo(f"Analysis complete for: {input_path}")
            typer.echo(f"Preset: {preset}")
            # TODO: Add formatted human-readable output
        
        sys.exit(ExitCode.SUCCESS)
        
    except ValidationError as e:
        typer.echo(f"Validation error: {e}", err=True)
        sys.exit(ExitCode.USER_ERROR)
    except Exception as e:
        typer.echo(f"Internal error during analysis: {e}", err=True)
        sys.exit(ExitCode.INTERNAL_ERROR)


@app.command()
def extend(
    file: str = typer.Argument(..., help="The input audio file to extend."),
    output: str = typer.Option(..., "--out", "-o", help="Output path for the extended audio file."),
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
    input_path = Path(file)
    output_path = Path(output)
    
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
        
        # TODO: Implement actual extension logic
        # This is a placeholder structure
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
            "warnings": [],
            "errors": [],
            "metrics": {
                "original_duration_s": 0.0,  # Placeholder
                "extended_duration_s": 0.0,  # Placeholder
                "bars_added": bars if bars else 0,  # Placeholder
                "mix_safety_score": 0.0,  # Placeholder
                "seam_quality": 0.0,  # Placeholder
            },
        }
        
        # Simulate refusal scenario (placeholder logic)
        # In real implementation, this would be based on actual analysis
        refused = False
        refusal_reason = None
        
        if refused:
            extension_result["status"] = "refused"
            extension_result["extended"] = False
            extension_result["refused"] = True
            extension_result["refusal_reason"] = refusal_reason or "Track not safe to extend"
            
            # Output refusal report
            if json_output or report:
                write_json_report(extension_result, report, pretty)
            else:
                typer.echo(f"Extension refused: {refusal_reason}", err=True)
            
            sys.exit(ExitCode.REFUSED)
        
        # Success case - generate output
        if not dry_run:
            # TODO: Actually generate the extended audio file
            # For now, just create the output directory structure
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Placeholder: would write actual audio file here
            typer.echo(f"Extended audio written to: {output_path}")
        
        # Output success report
        if json_output or report:
            write_json_report(extension_result, report, pretty)
        else:
            typer.echo(f"Extension complete: {input_path} -> {output_path}")
            if bars:
                typer.echo(f"Added {bars} bars")
            if seconds:
                typer.echo(f"Added {seconds} seconds")
        
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
