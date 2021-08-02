# ARp and Ipopt
# e = 1e-6
# 35 problems of MGH
using Printf

function solve_criteria(fm, fmin, e = 1e-6)
    if (fm - fmin) / max(1, abs(fmin)) <= e
        return 1
    else
        return 0
    end
end

function check_if_solved(inputFile)
    arpSolved = zeros(Float64, 35)
    ipoptSolved = zeros(Float64, 35)

    io = open(inputFile, "r")
    readline(io) # skip first line
    for i = 1:34
        # reading and parsing
        name, arp, ipopt = split(readline(io))
        arp = parse(Float64, arp)
        ipopt = parse(Float64, ipopt)

        fmin = min(arp, ipopt)

        arpSolved[i] = solve_criteria(arp, fmin)
        ipoptSolved[i] = solve_criteria(ipopt, fmin)
        @printf("%-10s %-10s %-10s\n",
                name, arpSolved[i] == 1.0 ? "y" : "n", ipoptSolved[i] == 1.0 ? "y" : "n")
    end
    close(io)

    arp = sum(arpSolved)
    ipopt = sum(ipoptSolved)
    println("ARp solved: $(arp)")
    println("Ipopt solved: $(ipopt)")
end

check_if_solved("Tests/outputTests.txt")
