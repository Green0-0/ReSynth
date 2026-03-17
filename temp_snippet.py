# Add reasoning option handling
        if enable_thinking:
            sampling_params.update({"thinking": True})
        else:
             # Ensure thinking is disabled if explicitly requested
             pass
