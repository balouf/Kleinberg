#   Kleinberg's Grid Simulator
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   The following code can be used to compute the Expected Delivery Time of the
#   Greedy Routing in Kleinberg's Grid.
# 
#   For more details, see the paper Kleinberg's Grid Reloaded
#   (https://hal.inria.fr/hal-01417096).
# 
#   Note : the present code possesses several improvements compared to the one
#   available in Kleinberg's Grid Reloaded (https://hal.inria.fr/hal-01417096):
# 
#     •  Arbitrary large computations are now feasible, thanks to
#        • Use of a double rejection sampling approach instead of a
#        single one
#        • Auto-selection of integer type between Int64, Int128 and
#        BigInt
# 
#     •  Multiple performance optimization
# 
#     •  Code is now embedded in a NoteBook for better readability

#   Usage: this notebook is designed to be included in another notebook. In the
#   first cell you evaluate
# 
#   using NBInclude
# 
#   nbinclude("KleinbergsGridSimulator.ipynb")
# 
#   This will give you access to the expecteddeliverytime function (an example
#   is provided in another NoteBook).

#   Additional remarks:
# 
#     •  It is very important that KleinbergsGridSimulator is the first
#        thing you load because it seems that using other packages
#        interfere with the compilation, resulting in a slowdown by
#        50%-200%.
# 
#     •  Tested on Julia 0.5; should also work on Julia 0.4 provided you
#        change \in by in in the proper places in the code.

#   Multi-Typing
#   ==============

#   It is vital that Julia knows that coordinates are integers so it can
#   pre-compile integer addition. Yet, depending on the value of n, we need to
#   operate on Int64, Int128, or even BigInt if one wants to compute the
#   universe. The following function will be used to provide Int64, Int128 and
#   BigInt versions of other functions. It uses Julia expressions.

function multitype(my_quoted_function, variable_list)
    my_types = [:Int64, :Int128, :BigInt]
    function explore(exp, my_type)
        if isa(exp, Expr)
            if length(exp.args) > 0
                # Type of monitored variables
                if (exp.head == :(::)) & (exp.args[1] ∈ variable_list)
                    exp.args[2] = my_type
                end
                # Conversion
                if (exp.head == :call) & (exp.args[1] ∈ my_types)
                    exp.args[1] = my_type
                end
            end
            # Recursion
            for i ∈ exp.args
                explore(i, my_type)
            end
        end
    end
    for typ ∈ my_types
        explore(my_quoted_function, typ)
        eval(my_quoted_function)
    end
end

#   Radius2Shortcut
#   =================

#   This function is just here to convert a radius into relative shortcut
#   coordinates. As it is specified as inlined, the redirection cost should be
#   nullified during compilation so we don't have to write it everywhere.

multitype(
quote
    @inline function radius2shortcut(radius::Int64)
        angle::Int64 = rand((-2*radius +1):(2*radius))
        return (radius - abs(angle)),  (sign(angle) * (radius - abs(radius - abs(angle))))
    end
end,
[:radius, :angle]
)

#   Radius Drawers
#   ================

#   The exact computation required to draw the radius depends on the value of r.
#   We'll basically prepare the 5 distinct cases independently, allowing the
#   compiler to be more optimal. The new radius drawers provide serious
#   performance gains compared with previous versions. Let N = 2(n-1) (called
#   max_radius in the code).

#   r Less than 1
#   –––––––––––––––

#   We want to use as covering function 1/x^{r-1}=x^{1-r} between 1 and N+1.
# 
#   What is the value of the sum already?
# 
#   \int_a^b x^{1-r}dx = 1/(2-r)(b^{2-r}-a^{2-r})

#   OK, let us now compute the reverse CDF so we can have a random generator
#   between 1 and N+1 that follows the correct distribution. Call z the random
#   uniform

# :$
# 
#   \int1^y x^{1-r}dx = z \int1^{N+1} x^{1-r}dx :$

# :$
# 
#   y^{2-r} -1 = z ( (N+1)^{2-r} -1 ) :$

# :$
# 
#   y = \sqrt[2-r]{z ( (N+1)^{2-r} -1 ) + 1} :$

#   Then we take the floor to have a nice k=\lfloor y \rfloor. Shall we take it?
#   Yes if we fall in a rectangle of height k^{1-r}
# 
# :$
# 
#   z/(2-r)((k+1)^{2-r}-k^{2-r}) < k^{1-r}:$
# 
# :$
# 
#   z((k+1)^{2-r}/k^{1-r}-k) < (2-r):$
# 
# :$
# 
#   z k((1+1/k)^{2-r}-1) < (2-r):$
# 
#   no otherwise.
# 
#   Let us turn that into code.

multitype(
quote
    @inline function Draw_r_smaller_than_1(n::Int64, r)
        expo = 2-r
        pow_max_radius = (2 * (n-1) + 1)^expo - 1
        function generator()
            radius = floor( (rand() * pow_max_radius + 1)^(1 / expo) )
            while rand() * radius * ((1 + 1 / radius)^expo - 1) > expo
                radius = floor( (rand() * pow_max_radius + 1)^(1 / expo) )
            end
            return radius2shortcut(Int64(radius))
        end
        return generator
    end
end,
[:n]
)

#   r Equal 1
#   –––––––––––

#   A simple uniform generator, plain and easy.

multitype(
quote
    @inline function Draw_r_equal_1(n::Int64)
        max_radius = 2*(n-1)
        return () -> radius2shortcut(rand(1:max_radius))
    end
end,
[:n]
)

#   r Between 1 and 2
#   –––––––––––––––––––

#   The function is now decreasing.
# 
#   To contain the (1/k^{r-1}) We use the following covering function:
# 
#     •  1 between 0 and 1
# 
#     •  1/x^{r-1}
#        between 1 and N

#   Probability to hit 1 with this function is
#   \frac{1}{1+\frac{N^{2-r}-1}{2-r}}.

#   Now, build the generator. Same computation that before yields
# 
# :$
# 
#   y = \sqrt[2-r]{z (N^{2-r} -1 ) + 1} :$

#   Then we take the ceil to have a nice k=\lceil y \rceil. Shall we take it?
#   Yes if
# 
# :$
# 
#   z/(2-r)(k^{2-r}-(k-1)^{2-r}) < k^{1-r}:$
# 
# :$
# 
#   z k(1-(1-1/k)^{2-r}) < (2-r):$
# 
#   no otherwise (that means full reset of the drawing).

multitype(
quote
    @inline function Draw_r_between_1_and_2(n::Int64, r)
        expo = 2-r
        pow_max_radius = (2 * (n-1))^expo - 1
        p1 = 1 / (1 + pow_max_radius / expo)
        un = Int64(1)
        function generator()
            while true
                if rand() < p1
                    return radius2shortcut(un)
                else
                    radius::Float64 = ceil( (rand() * pow_max_radius + 1)^(1 / expo) )
                    if rand() * radius * (1 - (1 - 1 / radius)^expo) < expo
                        return radius2shortcut(Int64(radius))
                    end
                end
            end
        end
        return generator
    end
end,
[:n]
)

#   r Equal 2
#   –––––––––––

#   We use the same covering function than before, but the sum between 1 and N
#   is just \log(N), so we have p_1 = 1/(1+\log(N)).

#   To draw a number, we first solve
# 
#   z\log(N) = \log(y)
# 
#   Hence
# 
#   y = N^z
# 
#   .

#   To accept:
# 
#   z\log(k/(k-1))<1/k
# 
#   zk\log(1+1/(k-1))<1

multitype(
quote
    @inline function Draw_r_equal_2(n::Int64)
        max_radius = 2*(n-1)
        p1 = 1/(1+log(max_radius))
        un = Int64(1)
        function generator()
            while true
                if rand() < p1
                    return radius2shortcut(un)
                else
                    radius::Float64 = ceil(max_radius^rand())
                    if rand()*radius*log(1+1/(radius-1)) < 1
                        return radius2shortcut(Int64(radius))
                    end
                end
            end
        end
        return generator
    end
end,
[:n]
)

#   r More than 2
#   –––––––––––––––

#   This is essentially the same thing than for 1<r<2, except that expo has now
#   opposite sign. To avoid confusing the compilation, let us rewrite this the
#   proper way.

#   The integral between 1 and N is \frac{1-N^{2-r}}{r-2}.

#   Probability to hit 1 with this function is
#   \frac{1}{1+\frac{1-N^{2-r}}{r-2}}.

#   Now, build the generator. Same computation that before yields
# 
# :$
# 
#   y = \sqrt[2-r]{z (N^{2-r} -1 ) + 1} :$

#   Then we take the ceil to have a nice k=\lceil y \rceil. Shall we take it?
#   Yes if
# 
# :$
# 
#   z/(r-2)((k-1)^{2-r}-k^{2-r}) < k^{1-r}:$
# 
# :$
# 
#   z k((1-1/k)^{2-r}-1) < (r-2):$
# 
# :$
# 
#   z k((1+1/(k-1))^{r-2}-1) < (r-2):$
# 
#   no otherwise (that means full reset of the drawing).

multitype(
quote
    @inline function Draw_r_greater_than_2(n::Int64, r)
        expo = r-2
        pow_max_radius = 1 / (2*(n-1))^expo - 1
        p1 = 1 / (1 - pow_max_radius / expo)
        un = Int64(1)
        function generator()
            while true
                if rand() < p1
                    return radius2shortcut(un)
                else
                    radius::Float64 = ceil( 1 / (rand() * pow_max_radius + 1)^(1 / expo))
                    if rand() * radius * ((1 + 1 / (radius - 1))^expo - 1) < expo
                        return radius2shortcut(Int64(radius))
                    end
                end
            end
        end
        return generator
    end
end,
[:n]
)

#   Main Grid Walking
#   ===================

#   The core algorithm with dynamic rejection sampling inside. Mostly untouched
#   since Kleinberg's Grid Reloaded (https://hal.inria.fr/hal-01417096), except
#   that it has been heavily typed to be sure that Julia JIT compiler does not
#   make any mistake.

multitype(
quote
    @inline function edt_gen(gen, n::Int64, p::Int64, q::Int64, R::Int64)    
        steps::Int64 = 0
        for i ∈ 1:R
            s_x::Int64, s_y::Int64, a_x::Int64, a_y::Int64 = tuple(rand(0:(n-1), 4)...)
            d = abs(s_x - a_x) + abs(s_y - a_y)
            while d>0
                d_s::Int64, sh_x::Int64, sh_y::Int64 = 2*n, -1, -1
                for j ∈ 1:q
                    c_s::Int64, ch_x::Int64, ch_y::Int64 = 2*n, -1, -1
                    while (ch_x < 0 || ch_x >= n || ch_y < 0 || ch_y >= n)
                        r_x::Int64, r_y::Int64 = gen()
                        ch_x, ch_y = s_x + r_x,  s_y + r_y
                    end
                    c_s = abs(a_x - ch_x) + abs(a_y - ch_y)
                    if c_s < d_s
                        d_s, sh_x, sh_y = c_s, ch_x, ch_y
                    end  
                end
                if d_s < d-p
                    d, s_x, s_y  = d_s, sh_x, sh_y
                else
                    d = d - p; Δx = min(p,abs(a_x - s_x)); Δy = p - Δx
                    s_x += Δx*sign(a_x - s_x); s_y += Δy*sign(a_y - s_y)
                end
                steps += 1    
            end
        end
        edt_value::Float64 = steps/R;    
        return edt_value
    end
end,
[:n, :s_x, :s_y, :a_x, :a_y, :d_s, :sh_x, :sh_y, :c_s, :ch_x, :ch_y, :r_x, :r_y]
)

#   Expected Delivery Time
#   ========================

#   We essentially put all pieces above together.
# 
#     •  Cast n properly (Int64/Int128/BigInt) to let the polymorphism
#        work. Note that the boundaries are 2^{61} and 2^{125}. The reason
#        is that we have 2(n-1) for the virtual ball, then 4i for the
#        angle. In the end, that's a 2^3-\epsilon factor in worst case
#        scenario. Also, we use signed integers as shortcuts can be
#        negative.
# 
#     •  Select the proper drawer.
# 
#     •  Launch core algorithm

function expected_delivery_time(n, r = 2, p = 1, q = 1, R = 10000, stats = true)
    starting_time = time()
    deux = Int128(2)
    if n ≤ 2^61
        n = Int64(n)
    elseif n ≤ deux^125
        n = Int128(n)
    else n = BigInt(n)
    end
    if r < 1
        gen = Draw_r_smaller_than_1(n, r)
    elseif r == 1
        gen = Draw_r_equal_1(n)
    elseif r < 2
        gen = Draw_r_between_1_and_2(n, r)
    elseif r == 2
        gen = Draw_r_equal_2(n)
    else
        gen = Draw_r_greater_than_2(n, r)
    end
    edt_value = edt_gen(gen, n, p, q, R)
    if stats
        duration = time() - starting_time
        println("Expected Delivery Time: $(round(edt_value, digits=2))")
        println("computed in $(round(duration, digits=2)) s")
        println("Parameters:  n = $n, r = $r, p = $p, q = $q, $R runs")
    end
    return edt_value
end

#   Legacy algorithms
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

#   Bulk
#   ======

#   This is essentially the version presented in Kleinberg's Grid Reloaded
#   (https://hal.inria.fr/hal-01417096), adapted to the common framework of this
#   version.

using StatsBase
@inline function BulkShortcut(n, r)
    bulk_size = min(n,2^23)
    probas = weights((1:(2*n-2)).^(1-Float64(r)))
    radii::Vector{Int64} = sample(1:(2*n-2), probas, bulk_size)
    index_radii::Int64 = 0
    function ShortcutDrawer()
        if index_radii == bulk_size
            sample!(1:(2*n-2), probas, radii)
            index_radii = 1
        end
        index_radii += 1
        return radius2shortcut(radii[index_radii])
    end
    return ShortcutDrawer
end

function expected_delivery_time_bulk(n, r = 2, p = 1, q = 1, R = 10000)
    gen = BulkShortcut(n, r)
    return edt_gen(gen, n, p, q, R)
end

#   Alias
#   =======

#   An intermediate, unpublished approach, based on Vose's Alias method
#   (http://www.keithschwarz.com/darts-dice-coins/). Better than Bulk, slightly
#   faster than using double rejection sampling for some parameters, but unable
#   to scale properly.

function InitiateAlias(n = 1000, r = 2)
  nprob = (1:(2*n-2)).^(1-Float64(r))
  nprob *= (2*n-2)/sum(nprob)
  Alias = zeros(Int64, (2*n-2))
  Prob = zeros(2*n-2)
  if r>=1
    thres = findfirst(nprob.<=1)
    # if isnothing(thres)
      # println(thres)
      # println(n)
      # println(r)
      # println(nprob)
    g = 1
    Small = 2*n-2
    l = 2*n-2
    while g<thres && Small>=thres
      Prob[l] = nprob[l]
      Alias[l] = g
      nprob[g] += nprob[l] - 1
      if nprob[g] < 1
        l = g
        g += 1
      else
        Small -= 1
        l = Small
      end
    end
  else
    thres = findfirst(nprob.>1)
    g = 2*n-2
    Small = 1
    l = 1
    while g>=thres && Small<thres
      Prob[l] = nprob[l]
      Alias[l] = g
      nprob[g] += nprob[l] - 1
      if nprob[g] < 1
        l = g
        g -= 1
      else
        Small += 1
        l = Small
      end
    end
  end
  Prob[Prob .== 0] .= 1
  return Prob, Alias
end

InitiateAlias(10,2);

@inline function AliasShortcut(n, r)
    Prob::Vector{Float64}, Alias::Vector{Int64} = InitiateAlias(n, r)
    function ShortcutDrawer()
        i = rand(1:(2*(n-1)))
        if rand()<Prob[i]
            return radius2shortcut(i)
        else
            return radius2shortcut(Alias[i])
        end
    end
    return ShortcutDrawer
end
AliasShortcut(10, 2);

function expected_delivery_time_alias(n = 1000, r = 2, p = 1, q = 1, R = 10000)
    gen = AliasShortcut(n, r)
    return edt_gen(gen, n, p, q, R)
end;