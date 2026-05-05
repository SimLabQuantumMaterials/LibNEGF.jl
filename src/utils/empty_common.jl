"""
	code_location()

Used when there is an error, where we want to exit graciously and indicate
where the problem has occurred.
"""
macro code_location()
    # return quote
    #     # just run the code itself, no timings
    #     $(esc(codex))
    # end
end