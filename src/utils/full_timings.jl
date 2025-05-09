using TimerOutputs

export TimingData

"""
	TimingData

When timing backend kernels within the most computationally demanding
inversions, this data allows us to enable or disable those finer-level
timings.
"""
struct TimingData
    to::TimerOutput
    label::String
end

"""
	timewrap(tdx, suffx, codex)

This macro allows us to enable or disable finer-level timings, where
the backend kernel calls are wrapped with it.
"""
macro timewrap(tdx, suffx, codex)
    return quote
        @timeit ($(esc(tdx))).to ($(esc(tdx))).label * ($(esc(suffx))) $(esc(codex))
    end
end