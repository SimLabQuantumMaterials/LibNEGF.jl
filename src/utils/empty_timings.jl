export TimingData

"""
	TimingData

The empty version of `TimingData` in `full_timings.jl`.
"""
struct TimingData
    # empty
end

struct CountingData
    # empty
end

"""
	TimingData

The non-timing version of `timewrap(tdx, suffx, codex)` in `full_timings.jl`.
"""
macro timewrap(tdx, cdx, suffx, codex)
    return quote
        # just run the code itself, no timings
        $(esc(codex))
    end
end