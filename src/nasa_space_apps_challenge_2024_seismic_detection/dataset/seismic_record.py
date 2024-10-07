import datetime
import os
import pickle
from pydantic import BaseModel
import numpy as np
from typing import Optional


class SeismicInterval(BaseModel):
    start_seconds: float
    # if start_idx is None, then the start of the interval is the start of the record
    start_idx: Optional[int] = None
    end_seconds: float
    # if end_idx is None, then the end of the interval is the end of the record
    end_idx: Optional[int] = None
    category: str


class SeismicRecord(BaseModel):
    record_id: str
    mseed_path: str
    space_body: str
    station: str

    markup: Optional[list[SeismicInterval]] = None

    waveform: np.ndarray

    # we likely do not need them here
    # because the more important to normalize data to the processing interval
    # instead of the whole record
    # spectrogram: list[list[float]]
    # mel: list[list[float]]

    class Config:
        arbitrary_types_allowed = True


def store_record(output_path: str, record: SeismicRecord):
    os.makedirs(output_path, exist_ok=True)
    dump_filename = os.path.join(output_path, f'{record.record_id}.pkl')
    with open(dump_filename, 'wb') as f:
        record_dump = record.model_dump()
        pickle.dump(record_dump, f)
    return dump_filename


def restore_record(filepath: str) -> SeismicRecord:
    with open(filepath, 'rb') as f:
        restored_record_dump = pickle.load(f)
        restored_record = SeismicRecord(**restored_record_dump)
    return restored_record


def list_records(output_path: str) -> list[str]:
    record_files = os.listdir(output_path)
    return record_files
