export TimingData

# a struct to pack timing info
# TODO : documentation
struct TimingData
    # empty
end

macro timewrap(tdx, suffx, codex)
    return quote
        # just run the code itself, no timings
        $(esc(codex))
    end
end