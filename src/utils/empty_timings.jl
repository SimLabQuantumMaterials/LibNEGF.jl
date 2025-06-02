export TimingData
export CountingData

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
macro timewrap(tdx, suffx, codex)
    return quote
        # just run the code itself, no timings
        $(esc(codex))
    end
end

macro countwrap(cdx, suffx, A, B, C, codex)
    return quote
        # just run the code itself, no timings
        $(esc(codex))
    end
end