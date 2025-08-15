"""
Parallelization strategies calculator.
"""


class ParallelizationCalculator:
    """Calculate memory distribution across parallelization strategies"""
    
    @staticmethod
    def calculate_tensor_parallel(memory: float, tp_size: int) -> float:
        """Calculate per-device memory with tensor parallelism"""
        return memory / tp_size
    
    @staticmethod
    def calculate_pipeline_parallel(memory: float, pp_size: int) -> float:
        """Calculate per-device memory with pipeline parallelism"""
        return memory / pp_size
    
    @staticmethod
    def calculate_expert_parallel(memory: float, ep_size: int, 
                                num_experts: int = 8) -> float:
        """Calculate per-device memory with expert parallelism"""
        # Assuming MoE model with expert parallelism
        expert_memory = memory * 0.8  # roughly 80% of memory is in experts
        shared_memory = memory * 0.2   # roughly 20% is shared
        
        return shared_memory + (expert_memory / ep_size)
    
    @staticmethod
    def calculate_data_parallel(memory: float, dp_size: int) -> float:
        """Calculate per-device memory with data parallelism"""
        # Each device holds full model copy
        return memory
    
    @staticmethod
    def calculate_combined_parallel(memory: float, tp_size: int = 1, pp_size: int = 1, 
                                  ep_size: int = 1, dp_size: int = 1) -> float:
        """Calculate per-device memory with combined parallelization strategies"""
        # Apply parallelization strategies in sequence
        # Note: this is a simplified calculation
        memory_after_tp = memory / tp_size if tp_size > 1 else memory
        memory_after_pp = memory_after_tp / pp_size if pp_size > 1 else memory_after_tp
        memory_after_ep = memory_after_pp / ep_size if ep_size > 1 else memory_after_pp
        # Data parallel doesn't reduce per-device memory
        
        return memory_after_ep
