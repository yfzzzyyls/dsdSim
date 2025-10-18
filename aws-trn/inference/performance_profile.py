import time
import os
import json
import csv
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """
    PerformanceProfiler encapsulates methods for measuring and recording:
      - Total generation latency
      - Per-token generation times
      - Token throughput (tokens per second)
      - Token match rate (for speculative decoding)
    It also provides methods to export these metrics to CSV and JSON files.
    """
    def __init__(self):
        self.start_time = None
        self.token_times = []
        self.token_count = 0
        self.match_count = 0  # For speculative mode
        self.total_latency = 0.0

    def start(self):
        self.start_time = time.time()
        self.token_times = []
        self.token_count = 0
        self.match_count = 0
        logger.debug("Performance profiling started.")

    def record_token(self, token_time: float, matched: bool = False):
        self.token_times.append(token_time)
        self.token_count += 1
        if matched:
            self.match_count += 1
        logger.debug(f"Recorded token {self.token_count}: {token_time:.4f} sec, matched={matched}")

    def finish(self):
        if self.start_time is not None:
            self.total_latency = time.time() - self.start_time
        logger.debug(f"Profiling finished. Total latency: {self.total_latency:.4f} sec")

    def average_token_time(self):
        if self.token_count > 0:
            return sum(self.token_times) / self.token_count
        return 0.0

    def throughput(self):
        if self.total_latency > 0:
            return self.token_count / self.total_latency
        return 0.0

    def token_match_rate(self):
        # For speculative decoding, the first token is from target only. 
        # If we have >1 tokens, we do (token_count - 1) in denominator
        if self.token_count > 1:
            return self.match_count / (self.token_count - 1)
        return None

    def export_metrics(self, role: str, output_dir: str = ".", filename_prefix: str = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = filename_prefix or role
        csv_filename = os.path.join(output_dir, f"{prefix}_performance_{timestamp}.csv")
        json_filename = os.path.join(output_dir, f"{prefix}_performance_{timestamp}.json")

        avg_time = self.average_token_time()
        tput = self.throughput()
        rate = self.token_match_rate()

        # CSV
        try:
            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["total_latency", "tokens_generated", "throughput", "avg_token_time", "token_match_rate"])
                writer.writerow([
                    f"{self.total_latency:.6f}",
                    self.token_count,
                    f"{tput:.6f}",
                    f"{avg_time:.6f}",
                    rate if rate is not None else "N/A"
                ])
            logger.info(f"Performance metrics saved to CSV: {csv_filename}")
        except Exception as e:
            logger.error(f"Failed to save CSV metrics: {e}")

        # JSON
        try:
            metrics = {
                "total_latency": self.total_latency,
                "tokens_generated": self.token_count,
                "throughput": tput,
                "avg_token_time": avg_time,
                "per_token_times": self.token_times,
                "token_match_rate": rate
            }
            with open(json_filename, "w") as jsonfile:
                json.dump(metrics, jsonfile, indent=2)
            logger.info(f"Performance metrics saved to JSON: {json_filename}")
        except Exception as e:
            logger.error(f"Failed to save JSON metrics: {e}")
