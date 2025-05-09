# taken from:
# https://discourse.julialang.org/t/how-to-print-function-name-and-source-file-line-number/43486/2
macro code_location()
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