"""
	ifdef(varx)

Mimicking `C`'s ifdef.
"""
macro ifdef(varx,codex)
    return quote
        if haskey(ENV,"LIBNEGF_HW")
            $(esc(codex))
        end
    end
end