using TimerOutputs

export TimingData

# a struct to pack timing info
# TODO : documentation
struct TimingData
    to::TimerOutput
    label::String
end

macro timewrap(tdx, suffx, codex)
    return quote
        @timeit ($(esc(tdx))).to ($(esc(tdx))).label * ($(esc(suffx))) $(esc(codex))
    end
end