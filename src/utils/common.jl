"""
	code_location()

Used when there is an error, where we want to exit graciously and indicate
where the problem has occurred.
"""
macro code_location()
    # taken from:
    # https://discourse.julialang.org/t/how-to-print-function-name-and-source-file-line-number/43486/2
    return quote
        st = stacktrace(backtrace())
        myf = ""
        for frm in st
            funcname = frm.func
            if frm.func != :backtrace && frm.func != Symbol("macro expansion")
                myf = frm.func
                break
            end
        end
        println("in function ", $("$(__module__)"), ".$(myf) at ", $("$(__source__.file)"), ":", $("$(__source__.line)"))
    end
end