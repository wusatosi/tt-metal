from math import ceil
import pandas as pd


class Chip:
    def __init__(
        self, name, peak_memory_bandwidth_gb, flops, freq, memory_capacity_gb, memory_efficiency, compute_efficiency
    ) -> None:
        self.name = name
        self.peak_memory_bandwidth_gb = peak_memory_bandwidth_gb
        self.flops = flops
        self.freq = freq
        self.memory_capacity_gb = memory_capacity_gb
        self.memory_efficiency = memory_efficiency
        self.compute_efficiency = compute_efficiency
        self.effective_gflops = self.flops * self.freq * self.compute_efficiency / 1e9
        self.effective_memory_bandwidth_GBps = self.peak_memory_bandwidth_gb * self.memory_efficiency


class System:
    def __init__(self, name, chip, num_instances) -> None:
        self.name = name
        self.chip = chip
        self.num_instances = num_instances
        self.effective_gflops = self.chip.effective_gflops * self.num_instances
        self.effective_memory_bandwidth_GBps = self.chip.effective_memory_bandwidth_GBps * self.num_instances
        self.memory_capacity_gb = self.chip.memory_capacity_gb * self.num_instances


class TransformerModel:
    def __init__(
        self,
        name,
        num_parameters_B,
        input_sequence_length,
        output_sequence_length,
        num_layers,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        intermediate_size,
        vocab_size,
    ) -> None:
        self.name = name
        self.num_parameters_B = num_parameters_B
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.head_dim = hidden_size // num_q_heads
        self.average_sequence_length = input_sequence_length + output_sequence_length / 2
        self.total_sequence_length = input_sequence_length + output_sequence_length

    def set_sequence_length(self, input_sequence_length, output_sequence_length):
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.average_sequence_length = input_sequence_length + output_sequence_length / 2
        self.total_sequence_length = input_sequence_length + output_sequence_length

    def model_size_B(self):
        attention_params = (
            self.hidden_size * self.hidden_size * ((self.num_q_heads + self.num_kv_heads) / self.num_q_heads + 1)
        )
        linear_params = self.hidden_size * self.hidden_size
        embedding_params = self.hidden_size * self.vocab_size
        if self.name.startswith("llama2") or self.name.startswith("llama3"):
            mlp_params = 3 * self.hidden_size * self.intermediate_size
            return (self.num_layers * (attention_params + linear_params + mlp_params) + embedding_params) / 1e9
        return 0

    def max_kv_cache_size_per_user_GB(self):
        return self.num_layers * self.total_sequence_length * self.num_kv_heads * self.head_dim * 2 / 1024**3

    def max_kv_cache_size_GB(self, num_users):
        return self.max_kv_cache_size_per_user_GB() * num_users

    def max_memory_size_GB(self, num_users):
        return self.max_kv_cache_size_GB(num_users) + self.num_parameters_B

    def avg_kv_cache_size_per_user_GB(self):
        return self.num_layers * self.average_sequence_length * self.num_kv_heads * self.head_dim * 2 / 1024**3

    def avg_kv_cache_size_GB(self, num_users):
        return self.avg_kv_cache_size_per_user_GB() * num_users

    def avg_memory_size_GB(self, num_users):
        return self.avg_kv_cache_size_GB(num_users) + self.num_parameters_B

    def max_num_users_that_fit_in_memory(self, system):
        if self.max_memory_size_GB(1) > system.memory_capacity_gb:
            return 0
        return (system.memory_capacity_gb - self.num_parameters_B) // self.max_kv_cache_size_per_user_GB()

    def dram_loading_mm_compute(self, row_size):
        return self.num_parameters_B * row_size * 2

    def attention_mm_compute(self, row_size):
        return self.num_layers * self.num_q_heads * self.head_dim * row_size * row_size * 2 * 2 / 1024**3

    # old version:
    # def decode_compute_gflops(self, num_users):
    #     return self.avg_memory_size_GB(1) * 2 * ceil(num_users / 32) * 32
    # new version:
    def decode_compute_gflops(self, num_users):
        return self.dram_loading_mm_compute(ceil(num_users / 32) * 32) + self.attention_mm_compute(
            self.average_sequence_length
        )

    def decode_compute_latency_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.decode_compute_gflops(num_users) / system.effective_gflops * 1000

    def decode_memory_latency_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.avg_memory_size_GB(num_users) / system.effective_memory_bandwidth_GBps * 1000

    def decode_latency_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return max(self.decode_compute_latency_ms(num_users, system), self.decode_memory_latency_ms(num_users, system))

    def decode_throughput_tokens_per_second_per_user(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return 1000 / self.decode_latency_ms(num_users, system)

    def decode_throughput_tokens_per_second(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.decode_throughput_tokens_per_second_per_user(num_users, system) * num_users

    def decode_total_time_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.decode_latency_ms(num_users, system) * self.output_sequence_length

    def prefill_compute_gflops(self):
        return self.dram_loading_mm_compute(self.input_sequence_length) + self.attention_mm_compute(
            self.input_sequence_length
        )

    def prefill_compute_latency_ms(self, system):
        return self.prefill_compute_gflops() / system.effective_gflops * 1000

    def prefill_memory_latency_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.decode_memory_latency_ms(num_users, system)

    def prefill_latency_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return max(self.prefill_compute_latency_ms(system), self.prefill_memory_latency_ms(num_users, system))

    def prefill_throughput_users_per_second(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return 1000 / self.prefill_latency_ms(num_users, system)

    def prefill_throughput_tokens_per_second(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.prefill_throughput_users_per_second(num_users, system) * self.input_sequence_length

    def prefill_total_time_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.prefill_latency_ms(num_users, system) * num_users

    def time_to_first_token_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.decode_latency_ms(num_users, system) + self.prefill_latency_ms(num_users, system)

    def time_to_last_token_ms(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.prefill_total_time_ms(num_users, system) + self.decode_total_time_ms(num_users, system)

    def overall_throughput_tokens_per_second_per_user(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.output_sequence_length * 1000 / self.time_to_last_token_ms(num_users, system)

    def overall_throughput_tokens_per_second(self, num_users, system):
        if self.max_num_users_that_fit_in_memory(system) < num_users:
            return -1
        return self.overall_throughput_tokens_per_second_per_user(num_users, system) * num_users

    def overall_throughput_at_max_num_users_tokens_per_second(self, system):
        return self.overall_throughput_tokens_per_second(self.max_num_users_that_fit_in_memory(system), system)

    def compute_all(self, num_users, system):
        return {
            "num_parameters(B)": self.num_parameters_B,
            "model_size(B)": self.model_size_B(),
            "max_kv_cache_size_per_user(GB)": self.max_kv_cache_size_per_user_GB(),
            "max_kv_cache_size(GB)": self.max_kv_cache_size_GB(num_users),
            "max_memory_size(GB)": self.max_memory_size_GB(num_users),
            "avg_kv_cache_size_per_user(GB)": self.avg_kv_cache_size_per_user_GB(),
            "avg_kv_cache_size(GB)": self.avg_kv_cache_size_GB(num_users),
            "avg_memory_size(GB)": self.avg_memory_size_GB(num_users),
            "max_num_users_that_fit_in_memory": self.max_num_users_that_fit_in_memory(system),
            "decode_compute(GFLOPS)": self.decode_compute_gflops(num_users),
            "decode_compute_latency(ms)": self.decode_compute_latency_ms(num_users, system),
            "decode_memory_latency(ms)": self.decode_memory_latency_ms(num_users, system),
            "decode_latency(ms)": self.decode_latency_ms(num_users, system),
            "decode_throughput(t/s/u)": self.decode_throughput_tokens_per_second_per_user(num_users, system),
            "decode_throughput(t/s)": self.decode_throughput_tokens_per_second(num_users, system),
            "decode_total_time(ms)": self.decode_total_time_ms(num_users, system),
            "prefill_compute(GFLOPS)": self.prefill_compute_gflops(),
            "prefill_compute_latency(ms)": self.prefill_compute_latency_ms(system),
            "prefill_memory_latency(ms)": self.prefill_memory_latency_ms(num_users, system),
            "prefill_latency(ms)": self.prefill_latency_ms(num_users, system),
            "prefill_throughput(u/s)": self.prefill_throughput_users_per_second(num_users, system),
            "prefill_throughput(t/s)": self.prefill_throughput_tokens_per_second(num_users, system),
            "prefill_total_time(ms)": self.prefill_total_time_ms(num_users, system),
            "time_to_first_token(ms)": self.time_to_first_token_ms(num_users, system),
            "time_to_last_token(ms)": self.time_to_last_token_ms(num_users, system),
            "overall_throughput(t/s/u)": self.overall_throughput_tokens_per_second_per_user(num_users, system),
            "overall_throughput(t/s)": self.overall_throughput_tokens_per_second(num_users, system),
            "overall_throughput_at_max_num_users(t/s)": self.overall_throughput_at_max_num_users_tokens_per_second(
                system
            ),
        }


def print_table(metric, column_names, row_names, table):
    # pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 1000)
    # pd.set_option("display.large_repr", 'info')
    # pd.set_option("display.expand_frame_repr", True)
    pd.set_option("display.width", 10000)
    pd.set_option("display.max_colwidth", 1000)
    # pd.set_option("display.precision", 5)

    print("=======================")
    print(f"{metric}")
    print("=======================")
    df = pd.DataFrame(table, columns=column_names, index=row_names)
    print(df)


# convert
# performance[model_name][system_name][input_sequence_length][output_sequence_length][metric] = value
# to
# new_performance[metric][system_name][model_name][input_sequence_length+output_sequence_length] = value
def convert_performance_layout(performance):
    new_performance = {}
    for model_name, systems in performance.items():
        for system_name, sequence_lengths in systems.items():
            for input_sequence_length, output_sequence_lengths in sequence_lengths.items():
                for output_sequence_length, metrics in output_sequence_lengths.items():
                    sequence_length = str(input_sequence_length) + "+" + str(output_sequence_length)
                    for metric, value in metrics.items():
                        if metric not in new_performance:
                            new_performance[metric] = {}
                        if system_name not in new_performance[metric]:
                            new_performance[metric][system_name] = {}
                        if model_name not in new_performance[metric][system_name]:
                            new_performance[metric][system_name][model_name] = {}
                        assert sequence_length not in new_performance[metric][system_name][model_name]
                        new_performance[metric][system_name][model_name][sequence_length] = value
    return new_performance


def print_performance(num_users, performance, metrics_to_print=set()):
    print("+++++++++++++++++++++")
    print(f"+++ num_users: {num_users} +++")
    print("+++++++++++++++++++++")
    for metric, systems in performance.items():
        if len(metrics_to_print) > 0 and metric not in metrics_to_print:
            continue
        column_names = []
        row_names = []
        rows = {}
        for system_name, models in systems.items():
            for model_name, sequence_lengths in models.items():
                column_names.append(system_name + "_" + model_name)
                for sequence_length, value in sequence_lengths.items():
                    if sequence_length not in rows:
                        row_names.append(sequence_length)
                        rows[sequence_length] = []
                    rows[sequence_length].append(value)
        table = []
        for sequence_length in row_names:
            table.append(rows[sequence_length])

        print_table(metric, column_names, row_names, table)


def main():
    # Define Chips
    WH = Chip(
        name="WH",
        peak_memory_bandwidth_gb=336,
        flops=8 * 16 * 16 * 64 * 2,
        freq=1e9,
        memory_capacity_gb=12,
        # memory_efficiency=0.8928,
        # compute_efficiency=0.60
        memory_efficiency=1.19,
        compute_efficiency=0.8,
    )

    # Create a BH chip
    BH = Chip(
        name="BH",
        peak_memory_bandwidth_gb=512,
        flops=8 * 16 * 16 * 140 * 2,
        freq=1e9,
        memory_capacity_gb=32,
        # memory_efficiency=0.75,
        # compute_efficiency=0.60
        memory_efficiency=1,
        compute_efficiency=0.8,
    )

    # Define Systems
    WH_Galaxy_x1 = System(name="WH_Galaxy_x1", chip=WH, num_instances=32)
    WH_Galaxy_x4 = System(name="WH_Galaxy_x4", chip=WH, num_instances=128)
    BH_Galaxy_x1 = System(name="BH_Galaxy_x1", chip=BH, num_instances=32)
    BH_Galaxy_x2 = System(name="BH_Galaxy_x2", chip=BH, num_instances=64)
    BH_Galaxy_x3 = System(name="BH_Galaxy_x3", chip=BH, num_instances=96)
    BH_Galaxy_x4 = System(name="BH_Galaxy_x4", chip=BH, num_instances=128)
    BH_Galaxy_x6 = System(name="BH_Galaxy_x6", chip=BH, num_instances=192)

    # Define Models
    llama3_8B = TransformerModel(
        name="llama3_8B",
        num_parameters_B=8,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=32,
        hidden_size=4096,
        num_q_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
    )

    llama3_70B = TransformerModel(
        name="llama3_70B",
        num_parameters_B=70,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=80,
        hidden_size=8192,
        num_q_heads=64,
        num_kv_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
    )

    llama3_400B = TransformerModel(
        name="llama3_400B",
        num_parameters_B=400,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=105,
        hidden_size=16384,
        num_q_heads=128,
        num_kv_heads=16,
        intermediate_size=65536,
        vocab_size=128256,
    )

    llama3_212B = TransformerModel(
        name="llama3_212B",
        num_parameters_B=212,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=80,
        hidden_size=16384,
        num_q_heads=64,
        num_kv_heads=8,
        intermediate_size=40960,
        vocab_size=128256,
    )

    llama3_1TB = TransformerModel(
        name="llama3_1TB",
        num_parameters_B=1024,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=100,
        hidden_size=32768,
        num_q_heads=64,
        num_kv_heads=8,
        intermediate_size=81920,
        vocab_size=128256,
    )

    # TODO:
    # 1. cache the computation between function calls
    # 2. remove sequence length from the model, add it to the calculation function
    # 3. add max_overall_thoughput_at_some_batch function (overall throughput drops sometimes due to prefill)

    # Build database for the performnace data
    num_users = 32
    performance = {}
    # for system in [WH_Galaxy_x1, BH_Galaxy_x1, BH_Galaxy_x2, BH_Galaxy_x3, BH_Galaxy_x4, BH_Galaxy_x6]:
    for system in [WH_Galaxy_x1, WH_Galaxy_x4]:
        # for model in [llama3_70B, llama3_212B, llama3_1TB]:
        for model in [llama3_8B, llama3_70B, llama3_400B]:
            # for input_sequence_length in [100, 1024, 7*1024, 31*1024, 199*1024]:
            for input_sequence_length in [128, 1024, 2048, 4096, 8192]:
                # output_sequence_length = 1024 if input_sequence_length > 100 else 100
                output_sequence_length = 1
                model.set_sequence_length(input_sequence_length, output_sequence_length)
                if model.name not in performance:
                    performance[model.name] = {}
                if system.name not in performance[model.name]:
                    performance[model.name][system.name] = {}
                if input_sequence_length not in performance[model.name][system.name]:
                    performance[model.name][system.name][input_sequence_length] = {}
                performance[model.name][system.name][input_sequence_length][output_sequence_length] = model.compute_all(
                    num_users, system
                )

    new_performance = convert_performance_layout(performance)

    # Print the performance data to stdout
    print_performance(
        num_users,
        new_performance,
        metrics_to_print={
            "prefill_latency(ms)",
            # 'decode_latency(ms)',
            "decode_throughput(t/s/u)",
            "time_to_first_token(ms)",
            # 'time_to_last_token(ms)',
            # 'overall_throughput(t/s/u)',
            # 'overall_throughput(t/s)',
            # 'max_num_users_that_fit_in_memory',
            # 'overall_throughput_at_max_num_users(t/s)'
        },
    )


if __name__ == "__main__":
    main()
