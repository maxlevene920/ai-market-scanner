#!/usr/bin/env python3
"""
AI Market Scanner - Main Entry Point

This is the main entry point for the AI Market Scanner application.
It provides a command-line interface for scanning markets and analyzing data.
"""

import asyncio
import click
from datetime import datetime, timedelta
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import settings
from src.scanner import MarketScanner
from src.agent import MarketAgent
from src.utils import setup_logging, get_logger, FileManager, DataProcessor


# Initialize console and logger
console = Console()
logger = get_logger("Main")


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--log-file', help='Log file path')
def cli(log_level: str, log_file: Optional[str]):
    """AI Market Scanner - Automated market analysis using AI agents"""
    setup_logging(log_level, log_file)
    logger.info("AI Market Scanner started")


@cli.command()
@click.option('--keywords', '-k', multiple=True, help='Keywords to search for')
@click.option('--max-results', '-m', default=200, help='Maximum number of results')
@click.option('--save', '-s', is_flag=True, help='Save results to file')
@click.option('--output-format', '-f', type=click.Choice(['json', 'csv']), default='json', help='Output format')
@click.option('--fantasy-football', '-ff', is_flag=True, help='Include fantasy football data sources')
def scan(keywords: List[str], max_results: int, save: bool, output_format: str, fantasy_football: bool):
    """Scan markets for mentions and save results"""
    
    # Use default keywords if none provided
    if not keywords:
        if fantasy_football:
            keywords = list(settings.scanner.fantasy_football_keywords)
        else:
            keywords = list(settings.scanner.keywords)
    
    console.print(f"[bold blue]Scanning markets for keywords: {', '.join(keywords)}[/bold blue]")
    
    async def run_scan():
        try:
            # Initialize scanner
            scanner = MarketScanner()
            
            # Test scanners first
            console.print("[yellow]Testing scanners...[/yellow]")
            test_results = await scanner.test_scanners()
            
            # Show test results
            table = Table(title="Scanner Test Results")
            table.add_column("Scanner", style="cyan")
            table.add_column("Status", style="green")
            
            for scanner_name, status in test_results.items():
                status_text = "✓ Working" if status else "✗ Failed"
                table.add_row(scanner_name, status_text)
            
            console.print(table)
            
            # Confirm if any scanners failed
            failed_scanners = [name for name, status in test_results.items() if not status]
            if failed_scanners:
                if not Confirm.ask(f"Some scanners failed: {', '.join(failed_scanners)}. Continue anyway?"):
                    console.print("[red]Scan cancelled by user[/red]")
                    return
            
            # Start scanning
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Scanning markets...", total=None)
                
                mentions = await scanner.scan_markets(keywords, max_results)
                
                progress.update(task, description="Scan completed!")
            
            # Display results
            console.print(f"\n[green]Found {len(mentions)} market mentions[/green]")
            
            if mentions:
                # Show top mentions
                table = Table(title="Top Market Mentions")
                table.add_column("Source", style="cyan")
                table.add_column("Title", style="white")
                table.add_column("Relevance", style="green")
                table.add_column("Published", style="yellow")
                
                for mention in mentions[:10]:
                    table.add_row(
                        mention.source,
                        mention.title[:50] + "..." if len(mention.title) > 50 else mention.title,
                        f"{mention.relevance_score:.2f}" if mention.relevance_score else "N/A",
                        mention.published_at.strftime("%Y-%m-%d %H:%M")
                    )
                
                console.print(table)
            
            # Save results if requested
            if save and mentions:
                file_manager = FileManager()
                
                if output_format == 'json':
                    file_path = file_manager.save_mentions(mentions)
                    console.print(f"[green]Results saved to: {file_path}[/green]")
                elif output_format == 'csv':
                    file_path = file_manager.export_to_csv(mentions)
                    console.print(f"[green]Results exported to: {file_path}[/green]")
            
        except Exception as e:
            logger.error(f"Error during scan: {e}")
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run_scan())


@cli.command()
@click.option('--keywords', '-k', multiple=True, help='Fantasy football keywords to search for')
@click.option('--max-results', '-m', default=100, help='Maximum number of results')
@click.option('--save', '-s', is_flag=True, help='Save results to file')
@click.option('--output-format', '-f', type=click.Choice(['json', 'csv']), default='json', help='Output format')
def fantasy_football(keywords: List[str], max_results: int, save: bool, output_format: str):
    """Scan fantasy football sources for mentions and analysis"""
    
    # Use fantasy football keywords if none provided
    if not keywords:
        keywords = list(settings.scanner.fantasy_football_keywords)
    
    console.print(f"[bold blue]Scanning fantasy football sources for keywords: {', '.join(keywords)}[/bold blue]")
    
    async def run_fantasy_scan():
        try:
            # Initialize scanner with fantasy football focus
            scanner = MarketScanner()
            
            # Test scanners first
            console.print("[yellow]Testing fantasy football scanners...[/yellow]")
            test_results = await scanner.test_scanners()
            
            # Show test results
            table = Table(title="Fantasy Football Scanner Test Results")
            table.add_column("Scanner", style="cyan")
            table.add_column("Status", style="green")
            
            for scanner_name, status in test_results.items():
                status_text = "✓ Working" if status else "✗ Failed"
                table.add_row(scanner_name, status_text)
            
            console.print(table)
            
            # Confirm if any scanners failed
            failed_scanners = [name for name, status in test_results.items() if not status]
            if failed_scanners:
                if not Confirm.ask(f"Some scanners failed: {', '.join(failed_scanners)}. Continue anyway?"):
                    console.print("[red]Scan cancelled by user[/red]")
                    return
            
            # Start scanning
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Scanning fantasy football sources...", total=None)
                
                mentions = await scanner.scan_markets(keywords, max_results)
                
                progress.update(task, description="Fantasy football scan completed!")
            
            # Display results
            console.print(f"\n[green]Found {len(mentions)} fantasy football mentions[/green]")
            
            if mentions:
                # Show top mentions
                table = Table(title="Top Fantasy Football Mentions")
                table.add_column("Source", style="cyan")
                table.add_column("Title", style="white")
                table.add_column("Relevance", style="green")
                table.add_column("Published", style="yellow")
                
                for mention in mentions[:10]:
                    table.add_row(
                        mention.source,
                        mention.title[:50] + "..." if len(mention.title) > 50 else mention.title,
                        f"{mention.relevance_score:.2f}" if mention.relevance_score else "N/A",
                        mention.published_at.strftime("%Y-%m-%d %H:%M")
                    )
                
                console.print(table)
            
            # Save results if requested
            if save and mentions:
                file_manager = FileManager()
                
                if output_format == 'json':
                    file_path = file_manager.save_mentions(mentions, prefix="fantasy_football")
                    console.print(f"[green]Fantasy football results saved to: {file_path}[/green]")
                elif output_format == 'csv':
                    file_path = file_manager.export_to_csv(mentions, prefix="fantasy_football")
                    console.print(f"[green]Fantasy football results exported to: {file_path}[/green]")
            
        except Exception as e:
            logger.error(f"Error during fantasy football scan: {e}")
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run_fantasy_scan())


@cli.command()
@click.option('--mentions-file', '-f', help='Mentions file to analyze')
@click.option('--keywords', '-k', multiple=True, help='Keywords to search for')
@click.option('--max-results', '-m', default=200, help='Maximum number of results')
@click.option('--analysis-type', '-t', type=click.Choice(['sentiment', 'trend', 'comprehensive']), 
              default='comprehensive', help='Type of analysis to perform')
@click.option('--save', '-s', is_flag=True, help='Save analysis results')
def analyze(mentions_file: Optional[str], keywords: List[str], max_results: int, 
           analysis_type: str, save: bool):
    """Analyze market mentions using AI agents"""
    
    console.print(f"[bold blue]Starting {analysis_type} analysis[/bold blue]")
    
    async def run_analysis():
        try:
            mentions = []
            
            if mentions_file:
                # Load mentions from file
                file_manager = FileManager()
                mentions = file_manager.load_mentions(mentions_file)
                
                if not mentions:
                    console.print(f"[red]No mentions found in file: {mentions_file}[/red]")
                    return
                
                console.print(f"[green]Loaded {len(mentions)} mentions from file[/green]")
            else:
                # Scan for new mentions
                if not keywords:
                    keywords = list(settings.scanner.keywords)
                
                console.print(f"[yellow]Scanning for keywords: {', '.join(keywords)}[/yellow]")
                
                scanner = MarketScanner()
                mentions = await scanner.scan_markets(keywords, max_results)
                
                if not mentions:
                    console.print("[red]No mentions found[/red]")
                    return
                
                console.print(f"[green]Found {len(mentions)} mentions[/green]")
            
            # Initialize appropriate agent
            if analysis_type == 'sentiment':
                from src.agent import SentimentAnalyzer
                agent = SentimentAnalyzer()
            elif analysis_type == 'trend':
                from src.agent import TrendAnalyzer
                agent = TrendAnalyzer()
            else:
                agent = MarketAgent()
            
            # Perform analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Performing {analysis_type} analysis...", total=None)
                
                result = await agent.analyze(mentions)
                
                progress.update(task, description="Analysis completed!")
            
            # Display results
            console.print(f"\n[green]Analysis completed with confidence: {result.confidence_score:.2f}[/green]")
            
            # Show insights
            if result.insights:
                console.print("\n[bold blue]Key Insights:[/bold blue]")
                for i, insight in enumerate(result.insights[:5], 1):
                    console.print(f"  {i}. {insight}")
            
            # Show recommendations
            if result.recommendations:
                console.print("\n[bold green]Recommendations:[/bold green]")
                for i, rec in enumerate(result.recommendations[:3], 1):
                    console.print(f"  {i}. {rec}")
            
            # Save results if requested
            if save:
                file_manager = FileManager()
                file_path = file_manager.save_analysis(result)
                console.print(f"[green]Analysis saved to: {file_path}[/green]")
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run_analysis())


@cli.command()
def status():
    """Show system status and configuration"""
    
    console.print("[bold blue]AI Market Scanner Status[/bold blue]")
    
    # Configuration status
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    config_table.add_column("Status", style="green")
    
    # Check API keys
    openai_status = "✓ Configured" if settings.api.openai_api_key else "✗ Missing"
    anthropic_status = "✓ Configured" if settings.api.anthropic_api_key else "✗ Missing"
    news_status = "✓ Configured" if settings.api.news_api_key else "✗ Missing"
    
    config_table.add_row("OpenAI API Key", "***" if settings.api.openai_api_key else "Not set", openai_status)
    config_table.add_row("Anthropic API Key", "***" if settings.api.anthropic_api_key else "Not set", anthropic_status)
    config_table.add_row("News API Key", "***" if settings.api.news_api_key else "Not set", news_status)
    config_table.add_row("Model Provider", settings.agent.model_provider, "✓")
    config_table.add_row("Model Name", settings.agent.model_name, "✓")
    config_table.add_row("Scan Interval", f"{settings.scanner.scan_interval_minutes} minutes", "✓")
    
    console.print(config_table)
    
    # Scanner status
    scanner = MarketScanner()
    scanner_status = scanner.get_scanner_status()
    
    scanner_table = Table(title="Available Scanners")
    scanner_table.add_column("Scanner", style="cyan")
    scanner_table.add_column("Rate Limit", style="yellow")
    scanner_table.add_column("Status", style="green")
    
    for scanner_info in scanner_status["scanners"]:
        scanner_table.add_row(
            scanner_info["name"],
            f"{scanner_info['rate_limit']} req/s",
            "✓ Enabled" if scanner_info["enabled"] else "✗ Disabled"
        )
    
    console.print(scanner_table)
    
    # File system status
    file_manager = FileManager()
    dir_info = file_manager.get_directory_size()
    
    fs_table = Table(title="File System")
    fs_table.add_column("Directory", style="cyan")
    fs_table.add_column("Size", style="yellow")
    fs_table.add_column("Files", style="green")
    
    fs_table.add_row("Total", f"{dir_info['total_size_mb']} MB", str(dir_info['total_files']))
    
    for dir_name, info in dir_info['directories'].items():
        fs_table.add_row(dir_name, f"{info['size_mb']} MB", str(info['file_count']))
    
    console.print(fs_table)


@cli.command()
@click.option('--days', '-d', default=30, help='Number of days to keep files')
def cleanup(days: int):
    """Clean up old files"""
    
    console.print(f"[yellow]Cleaning up files older than {days} days...[/yellow]")
    
    file_manager = FileManager()
    deleted_count = file_manager.cleanup_old_files(days)
    
    console.print(f"[green]Cleaned up {deleted_count} old files[/green]")


@cli.command()
def list_files():
    """List saved mentions and analysis files"""
    
    file_manager = FileManager()
    
    # List mentions files
    mentions_files = file_manager.list_mentions_files()
    if mentions_files:
        console.print("\n[bold blue]Mentions Files:[/bold blue]")
        for filename in mentions_files[:10]:  # Show first 10
            file_info = file_manager.get_file_info("mentions", filename)
            if file_info:
                console.print(f"  {filename} ({file_info['size_mb']} MB, {file_info['modified']})")
    else:
        console.print("\n[yellow]No mentions files found[/yellow]")
    
    # List analysis files
    analysis_files = file_manager.list_analysis_files()
    if analysis_files:
        console.print("\n[bold blue]Analysis Files:[/bold blue]")
        for filename in analysis_files[:10]:  # Show first 10
            file_info = file_manager.get_file_info("analysis", filename)
            if file_info:
                console.print(f"  {filename} ({file_info['size_mb']} MB, {file_info['modified']})")
    else:
        console.print("\n[yellow]No analysis files found[/yellow]")


if __name__ == "__main__":
    cli()
