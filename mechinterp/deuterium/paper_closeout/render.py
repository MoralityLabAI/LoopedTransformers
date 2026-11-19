"""Build the completed Deuterium report without rewriting historical outputs.

The PDF layout is derived from the frozen retrospective renderer. This separate
entry point preserves its source, generated blocks, and archived PDF byte for
byte. Rendering and receipt verification never import Torch or generate data.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import statistics
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'papers' / 'VPD_TRM_Feature_Steering'
STEM = 'deuterium_complete_20261119'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipts() -> dict[str, Any]:
    """Check exact archive-member hashes before using recorded results."""
    checked = {}
    for manifest in sorted((PAPER / 'evidence').glob('*/provenance.json')):
        source = json.loads(manifest.read_text())
        for name, record in source['files'].items():
            path = manifest.parent / name
            if not path.is_relative_to(PAPER / 'evidence') or '..' in Path(name).parts:
                raise ValueError(f'Unsafe evidence path: {name}')
            if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise ValueError(f'Archived receipt changed: {path}')
            checked[str(path.relative_to(ROOT))] = record['sha256']
    if not checked:
        raise ValueError('No archived evidence receipts found')
    gate = json.loads((PAPER / 'evidence/hosted_gate/paper_summary.json').read_text())
    native = json.loads((PAPER / 'evidence/native01/registered_verdict.json').read_text())
    if gate['status'] != 'GitHub Actions: NO_REGISTERED_ADVANTAGE':
        raise ValueError('Unexpected hosted-gate conclusion')
    if native['status'] != 'DEVELOPMENT_NO_REGISTERED_ADVANTAGE':
        raise ValueError('Unexpected native steering conclusion')
    if native['teacher_seeds'] != [103, 211, 307]:
        raise ValueError('Native teacher roster changed')
    for record in native['per_teacher_audits'].values():
        if record['status'] != 'PASS' or record['errors']:
            raise ValueError('Native independent audit did not pass')
    effects = [float(x['effect']) for x in gate['primary_effects']]
    if abs(statistics.mean(effects) - gate['aggregate_effect']['effect']) > 1e-14:
        raise ValueError('Hosted per-root effect arithmetic disagrees')
    if abs(statistics.mean(native['primary_by_old_teacher'].values()) - native['primary_mean']) > 1e-14:
        raise ValueError('Native per-root effect arithmetic disagrees')
    schedule = ROOT / 'native_trm/adapter02/results/source_schedule_20261119'
    provenance = json.loads((schedule / 'provenance.json').read_text())
    portable = provenance['portable_trace_archive']
    archive = schedule / portable['file']
    if archive.name != 'portable_optimizer_traces.zip' or sha256(archive) != portable['sha256']:
        raise ValueError('Original source-only optimizer trace archive changed')
    from mechinterp.deuterium.native_trm.adapter02.audit_schedule import audit
    with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(archive) as packed:
        if sorted(packed.namelist()) != portable['members']:
            raise ValueError('Source-only optimizer archive membership changed')
        for name in packed.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe source-only optimizer archive member')
        packed.extractall(folder)
        optimizer = audit(Path(folder))
    if optimizer['parity_status'] != 'BLOCKED_UNMATCHED_OPTIMIZER':
        raise ValueError('Archived optimizer verdict disagrees with the completed paper')
    return {'files_checked': len(checked), 'receipt_sha256': checked,
            'hosted_gate': gate, 'native01': native, 'optimizer_audit': optimizer}


def completed_effects(evidence: dict[str, Any]) -> list[Path]:
    """Plot two completed studies with separate endpoint scales."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.hashsalt': 'deuterium-completed-20261119',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False})
    gate, native = evidence['hosted_gate'], evidence['native01']
    panels = [
        ('1,763-parameter recurrent gate', 'Output-head edit; generated commit/veto/repair task',
         [r['effect'] for r in gate['primary_effects']],
         gate['aggregate_effect']['effect'], gate['aggregate_effect']['ci95']),
        ('44,001-parameter four-pass recurrent model', 'Shared-block edit; cyclic-copy task; previously examined teachers',
         list(native['primary_by_old_teacher'].values()),
         native['primary_mean'], native['primary_ci95']),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6))
    for ax, (title, subtitle, effects, mean, interval) in zip(axes, panels):
        ax.axvline(0, color='#84959e', linewidth=1, linestyle='--', zorder=0)
        ax.scatter(effects, [3, 2, 1], color='#436f86', s=35, zorder=3)
        ax.errorbar(mean, 0, xerr=[[mean - interval[0]], [interval[1] - mean]],
                    fmt='D', color='#bd622c', markersize=5, capsize=4, linewidth=1.7)
        ax.set_yticks([3, 2, 1, 0], ['Seed 103', 'Seed 211', 'Seed 307', 'Mean + 95% CI'])
        ax.set_ylim(-0.6, 3.6)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.4f}'))
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=25, color='#173d52')
        ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=8, color='#526670')
        ax.set_xlabel('Random minus learned-VPD target NLL (nats)', fontsize=9)
        ax.grid(axis='x', color='#e3e9eb', linewidth=.6, zorder=0)
        lo, hi = min(interval[0], *effects), max(interval[1], *effects)
        padding = (hi - lo) * .12
        ax.set_xlim(lo - padding, hi + padding)
    fig.subplots_adjust(left=.22, right=.98, top=.89, bottom=.1, hspace=.8)
    dest = PAPER / 'figures/completed_steering_effects'
    paths = [dest.with_suffix('.png'), dest.with_suffix('.svg')]
    fig.savefig(paths[0], dpi=220, metadata={'Software': 'Deuterium completed-study evidence'})
    fig.savefig(paths[1], metadata={'Date': None, 'Creator': 'Deuterium completed-study evidence'})
    paths[1].write_text('\n'.join(line.rstrip() for line in paths[1].read_text().splitlines()) + '\n')
    plt.close(fig)
    return paths


def pdf_from_markdown(md: Path, dest: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                  TableStyle, Image, Preformatted)
    import matplotlib
    fontroot = Path(matplotlib.get_data_path())/'fonts'/'ttf'
    # Bundled TrueType fonts keep long paths and Unicode metadata legible.
    if not fontroot.exists():
        raise FileNotFoundError('Pinned matplotlib DejaVu font bundle is missing.')
    for name, filename in [('PaperSerif', 'DejaVuSerif.ttf'), ('PaperSerifBold', 'DejaVuSerif-Bold.ttf'),
                           ('PaperSerifItalic', 'DejaVuSerif-Italic.ttf'),
                           ('PaperSans', 'DejaVuSans.ttf'), ('PaperSansBold', 'DejaVuSans-Bold.ttf'),
                           ('PaperMono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fontroot / filename)))
    pdfmetrics.registerFontFamily('PaperSerif', normal='PaperSerif', bold='PaperSerifBold',
                                  italic='PaperSerifItalic', boldItalic='PaperSerifBold')
    pdfmetrics.registerFontFamily('PaperSans', normal='PaperSans', bold='PaperSansBold',
                                  italic='PaperSans', boldItalic='PaperSansBold')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyPaper', fontName='PaperSerif', fontSize=9.4, leading=13.4,
                              textColor=colors.HexColor('#20292e'), spaceAfter=7.5,
                              allowWidows=0, allowOrphans=0, splitLongWords=True))
    styles.add(ParagraphStyle('TitlePaper', fontName='PaperSansBold', fontSize=21, leading=26,
                              textColor=colors.HexColor('#173d52'), spaceAfter=9))
    styles.add(ParagraphStyle('SubtitlePaper', fontName='PaperSerifItalic', fontSize=13.5, leading=18,
                              textColor=colors.HexColor('#475b66'), spaceAfter=14))
    styles.add(ParagraphStyle('SectionPaper', fontName='PaperSansBold', fontSize=12.5, leading=17,
                              textColor=colors.HexColor('#173d52'), spaceBefore=13, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle('SubsectionPaper', fontName='PaperSansBold', fontSize=10.5, leading=15,
                              textColor=colors.HexColor('#173d52'), spaceBefore=10, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle('CaptionPaper', parent=styles['BodyPaper'], fontName='PaperSerifItalic',
                              fontSize=8.2, leading=11.5, textColor=colors.HexColor('#536570')))
    styles.add(ParagraphStyle('TablePaper', fontName='PaperSans', fontSize=7.3, leading=10,
                              textColor=colors.HexColor('#20292e'), splitLongWords=True))
    styles.add(ParagraphStyle('TableHeadPaper', parent=styles['TablePaper'], fontName='PaperSansBold',
                              textColor=colors.white))
    styles.add(ParagraphStyle('CodePaper', fontName='PaperMono', fontSize=6.8, leading=9,
                              textColor=colors.HexColor('#33474f'), spaceAfter=7))
    width = 7.0 * inch
    doc = SimpleDocTemplate(str(dest), pagesize=(8.5*inch, 11*inch), rightMargin=.75*inch,
                            leftMargin=.75*inch, topMargin=.65*inch, bottomMargin=.65*inch,
                            title='Deuterium: Auditing Parameter Steering in Recurrent Models', author='Patrick Dugan / MoralityLabAI',
                            subject='Completed retrospective audit and native recurrent steering report')
    def inline(s: str) -> str:
        s = html.escape(s)
        # Convert portable Markdown inline constructs, leaving exact code text visible.
        s = re.sub(r'`([^`]+)`', lambda m: '<font face="PaperMono" size="7.6">'+m.group(1)+'</font>', s)
        s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
        s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', s)
        def link(m: re.Match) -> str:
            label, url = m.group(1), html.unescape(m.group(2))
            if url.startswith(('https://', 'http://')):
                return f'<a href="{html.escape(url, quote=True)}" color="#225f7d">{label}</a>'
            return label
        s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, s)
        return s
    story: list[Any] = []
    lines = md.read_text().splitlines()
    i = 0
    subtitle_used = False
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('<!--'):
            i += 1; continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i]); i += 1
            story.append(Preformatted('\n'.join(code), styles['CodePaper'], maxLineLength=93))
            i += 1; continue
        image_match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image_match:
            path = md.parent / image_match.group(2)
            pic = Image(str(path))
            height = width * pic.imageHeight / pic.imageWidth
            pic.drawWidth, pic.drawHeight = width, height
            pic.keepWithNext = True
            after_image = Spacer(1, 3)
            after_image.keepWithNext = True
            story += [Spacer(1, 5), pic, after_image]
            i += 1; continue
        if line.startswith('|'):
            matrix = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in re.split(r'(?<!\\)\|', lines[i].strip().strip('|'))]
                if not all(re.fullmatch(r'[-: ]+', c) for c in cells): matrix.append(cells)
                i += 1
            cols = len(matrix[0])
            # Relative content lengths guide widths with a floor for readable numeric columns.
            weight = []
            for j in range(cols):
                lengths = sorted(min(len(re.sub(r'`|\*', '', row[j])), 70) for row in matrix if j < len(row))
                avg = sum(lengths)/len(lengths)
                weight.append(max(9., min(42., avg)))
            col_widths = [width*w/sum(weight) for w in weight]
            data = [[Paragraph(inline(c), styles['TableHeadPaper' if r == 0 else 'TablePaper'])
                     for c in row] for r, row in enumerate(matrix)]
            tab = Table(data, colWidths=col_widths, repeatRows=1, hAlign='LEFT',
                        rowSplitRange=(3, -1) if len(data) > 3 else None)
            tab.spaceBefore, tab.spaceAfter = 4, 9
            tab.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#224c63')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f1f5f6'), colors.white]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LINEBELOW', (0,-1), (-1,-1), .4, colors.HexColor('#bbcbd2')),
            ]))
            story.append(tab)
            continue
        heading = re.match(r'^(#{1,3}) (.*)$', line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2)
            if level == 1: style = 'TitlePaper'
            elif level == 2 and not subtitle_used:
                style = 'SubtitlePaper'; subtitle_used = True
            else: style = 'SectionPaper' if level == 2 else 'SubsectionPaper'
            story.append(Paragraph(inline(title), styles[style]))
            i += 1; continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '|', '```', '<!--', '![')):
            para.append(lines[i].strip()); i += 1
        text = ' '.join(para)
        style = styles['CaptionPaper'] if text.startswith('*Figure ') else styles['BodyPaper']
        story.append(Paragraph(inline(text), style))
    def page(canvas_: Any, doc_: Any) -> None:
        canvas_.saveState()
        canvas_.setFont('PaperSans', 7)
        canvas_.setFillColor(colors.HexColor('#667d88'))
        canvas_.drawString(.75*inch, .35*inch, 'DEUTERIUM  |  Retrospective audit and recurrent steering  |  19 November 2026')
        canvas_.drawRightString(7.75*inch, .35*inch, str(doc_.page))
        canvas_.restoreState()
    def fixed_canvas(*args: Any, **kwargs: Any) -> Any:
        kwargs['invariant'] = 1
        return canvas.Canvas(*args, **kwargs)
    doc.build(story, onFirstPage=page, onLaterPages=page, canvasmaker=fixed_canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify saved receipt and render hashes without changing files')
    args = parser.parse_args()
    evidence = verify_receipts()
    manifest_path = PAPER / (STEM + '_manifest.json')
    if args.verify_only:
        saved = json.loads(manifest_path.read_text())
        for category in ('inputs', 'outputs'):
            for name, expected in saved[category].items():
                if sha256(ROOT / name) != expected:
                    raise ValueError(f'Closeout {category} hash mismatch: {name}')
        print(json.dumps({'status': 'PASS', 'receipts_checked': evidence['files_checked'],
                          'render_outputs_checked': len(saved['outputs']),
                          'model_inference_performed': False}, sort_keys=True))
        return
    paper = PAPER / (STEM + '.md')
    text = paper.read_text()
    if any(token in text for token in ('TODO', 'TBD', 'GitHub Actions reproduction awaits')):
        raise ValueError('Unresolved manuscript placeholder or superseded execution status')
    figures = completed_effects(evidence)
    pdf = PAPER / (STEM + '.pdf')
    pdf_from_markdown(paper, pdf)
    sources = [paper, Path(__file__).resolve()]
    sources += sorted((PAPER / 'evidence').rglob('provenance.json'))
    sources += [ROOT / name for name in evidence['receipt_sha256']]
    sources += [PAPER / 'COMPLETION_AND_SILICO.md']
    sources += sorted((ROOT / 'native_trm/adapter02/results/source_schedule_20261119').glob('*'))
    sources += [ROOT / 'native_trm/adapter02/audit_schedule.py']
    sources += [PAPER / 'figures' / name for name in
                ('arc_overlap.png', 'followup_effects.png', 'replication_by_seed.png')]
    outputs = figures + [pdf]
    manifest = {
        'schema': 'deuterium-completed-paper-render-v1',
        'report_date': '2026-11-19',
        'historical_artifacts_modified': False,
        'model_inference_performed': False,
        'new_scientific_data_generated': False,
        'receipt_files_verified': evidence['files_checked'],
        'source_only_optimizer_measurement': evidence['optimizer_audit'],
        'inputs': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'outputs': {str(p.relative_to(ROOT)): sha256(p) for p in outputs},
    }
    import matplotlib
    fontroot = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    manifest['font_sha256'] = {p.name: sha256(p) for p in sorted(fontroot.glob('DejaVu*.ttf'))}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'PASS', 'paper': str(paper.relative_to(ROOT)),
                      'pdf': str(pdf.relative_to(ROOT)),
                      'receipts_checked': evidence['files_checked']}, sort_keys=True))


if __name__ == '__main__':
    main()
