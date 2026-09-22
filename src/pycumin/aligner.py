import os
import sys
from dataclasses import dataclass
from http.client import responses
from pathlib import Path
from typing import Optional, Any, Iterable

import attrs

from . import Index, Alignment


@dataclass
class Alignment:
    ctg: str
    r_st: int
    r_en: int
    strand: int

@attrs.define
class Result:
    """Result holder

    This should be progressively filled with data from the basecaller,
    barcoder, and then the aligner.

    :param channel: The channel that this read is being sequenced on
    :param read_id: The read ID assigned to this read by MinKNOW
    :param seq: The basecalled sequence for this read
    :param decision: The ``Decision`` that has been made, this will by used to determine the ``Action``
    :param barcode: The barcode that has been assigned to this read
    :param basecall_data: Any extra data that the basecaller may want to send to the aligner
    :param alignment_data: Any extra alignment data
    """

    channel: int
    read_id: str
    seq: str
    barcode: Optional[str] = attrs.field(default=None)
    basecall_data: Optional[Any] = attrs.field(default=None)
    alignment_data: Optional[list[Alignment]] = attrs.field(default=None)

class Aligner:

    def __init__(self, debug_log: str | None = None, **kwargs):
        if debug_log:
            if debug_log == 'stdout':
                self.logfile = sys.stdout
            elif debug_log == 'stderr':
                self.logfile = sys.stderr
            else:
                self.logfile = open(debug_log, 'w')
        else:
            self.logfile = None
        self.kwargs = kwargs
        if 'input' not in kwargs:
            raise AttributeError('Required argument "input" not found.')
        else:
            self.input = kwargs['input']
        self.threshold = kwargs.get('threshold', 0.15)
        self.aligner = Index(self.input)

    def validate(self) -> None:
        input: str = self.kwargs["input"]
        file_extensions = [".npx"]
        if all((not Path(input).is_file(), input)):
            raise FileNotFoundError(f"{input} does not exist")
        if not any(input.lower().endswith(suffix) for suffix in file_extensions):
            raise RuntimeError(
                f"Provided index file appears to be of an incorrect type - should be one of {file_extensions}"
            )

    @property
    def initialised(self) -> bool:
        return True

    def describe(self, regions: list, barcodes: dict) -> str:
        print(f"loaded {self.input}: {self.aligner.n_entries:,} entries, {self.aligner.nbytes() / 1e9:.2f} GB, {self.aligner.p}")
        return ("Indexed reference file {}. Hopefully, I will return a more meaningful description in the future".
                format(self.kwargs['input']))

    def map_reads(self, calls: Iterable[Result]) -> Iterable[Result]:
        for result in calls:
            seq = result.seq
            if not seq:
                result.alignment_data = []
            else:
                s_, wid, strand, nq = self.aligner.query(seq)
                ref, ws = self.aligner.window_locus(wid) if wid >= 0 else ("*", -1)
                if s_ > self.threshold:
                    result.alignment_data = [Alignment(ref, ws, ws + len(seq), strand)]
                else:
                    result.alignment_data = []
            yield result

    def disconnect(self):
        if self.logfile and self.logfile not in [sys.stdout, sys.stderr]:
            self.logfile.close()