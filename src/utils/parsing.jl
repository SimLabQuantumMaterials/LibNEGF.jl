"""
	ifdef(varx)

Mimicking `C`'s ifdef.
"""
macro ifdef(varx,codex)
    return quote
        if haskey(ENV,$(esc(varx)))
            $(esc(codex))
        end
    end
end