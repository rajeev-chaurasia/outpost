"""Unit tests for MappingReport's filtering."""

from outpost.mapping import MappingEntry, MappingOutcome, MappingReport


def _entry(outcome: MappingOutcome) -> MappingEntry:
    return MappingEntry(
        source_id="widgets", row=1, field="widget_id", outcome=outcome, raw_value="w-1"
    )


def test_report_filters_entries_by_outcome() -> None:
    report = MappingReport()
    report.add(_entry(MappingOutcome.MAPPED))
    report.add(_entry(MappingOutcome.NEEDS_REVIEW))
    report.add(_entry(MappingOutcome.UNMAPPED))
    report.add(_entry(MappingOutcome.MAPPED))

    assert len(report.mapped()) == 2
    assert len(report.needs_review()) == 1
    assert len(report.unmapped()) == 1
    assert len(report.entries) == 4
