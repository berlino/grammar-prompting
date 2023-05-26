%:- ensure_loaded(geoquery).

:- style_check(-singleton).
:- style_check(-discontiguous).

%eval([]).
%eval([I,J,F1,F2|L]) :-
%   execute_funql_query(F1, A1),
%   execute_funql_query(F2, A2),
%   print(A1), print(' '), print(J), (A1 == A2 -> print(' y') ; print(' n')), nl,
%   eval(L).

%eval([]).
%eval([F1|L]) :-
%   execute_funql_query(F1, A1),
%   print(A1), nl,
%   eval(L).

eval(F1, A1) :-
   execute_funql_query(F1, A1).

execute_funql_query(null, null).
execute_funql_query(Q, U) :- process(Q,P), sort(P, U).
execute_funql_query(Q, []). % empty result

process(answer(Q), P) :- process(Q, P).

process(stateid(A), [stateid(A)]).
process(cityid(A,B), [cityid(A,B)]).
process(riverid(A), [riverid(A)]).
process(countryid(A), [countryid(A)]).
process(placeid(A), [placeid(A)]).

process(city(all), A) :- findall(B, city(B), A).
process(mountain(all), A) :- findall(B, place(B), A).
process(place(all), A) :- findall(B, place(B), A).
process(river(all), A) :- findall(B, river(B), A).
process(lake(all), A) :- findall(B, lake(B), A).
process(state(all), A) :- findall(B, state(B), A).
process(capital(all), A) :- findall(B, capital(B), A).

% filter the list by the predicate
process(capital(A), P) :- process(A,L), process(capital(L), P).
process(capital([]), []).
process(capital([A|AA]), [A|PP]) :- capital(A), !, process(capital(AA), PP).
process(capital([A|AA]), PP) :- process(capital(AA), PP).
process2(capital(A), P) :- process2(A,L), process2(capital(L), P).
process2(capital([]), []).
process2(capital([A-S|AA]), [PA-S|PP]) :- process(capital(A),PA), process2(capital(AA), PP).

process(city(A), P) :- process(A,L), process(city(L), P).
process(city([]), []).
process(city([A|AA]), [A|PP]) :- city(A), !, process(city(AA), PP).
process(city([A|AA]), PP) :- process(city(AA), PP).
process2(city(A), P) :- process2(A,L), process2(city(L), P).
process2(city([]), []).
process2(city([A-S|AA]), [PA-S|PP]) :- process(city(A),PA), process2(city(AA), PP).

process(major(A), P) :- process(A,L), process(major(L), P).
process(major([]), []).
process(major([A|AA]), [A|PP]) :- major(A), !, process(major(AA), PP).
process(major([A|AA]), PP) :- process(major(AA), PP).
process2(major(A), P) :- process2(A,L), process2(major(L), P).
process2(major([]), []).
process2(major([A-S|AA]), [PA-S|PP]) :- process(major(A),PA), process2(major(AA), PP).

process(place(A), P) :- process(A,L), process(place(L), P).
process(place([]), []).
process(place([A|AA]), [A|PP]) :- place(A), !, process(place(AA), PP).
process(place([A|AA]), PP) :- process(place(AA), PP).
process2(place(A), P) :- process2(A,L), process2(place(L), P).
process2(place([]), []).
process2(place([A-S|AA]), [PA-S|PP]) :- process(place(A),PA), process2(place(AA), PP).

process(river(A), P) :- process(A,L), process(river(L), P).
process(river([]), []).
process(river([A|AA]), [A|PP]) :- river(A), !, process(river(AA), PP).
process(river([A|AA]), PP) :- process(river(AA), PP).
process2(river(A), P) :- process2(A,L), process2(river(L), P).
process2(river([]), []).
process2(river([A-S|AA]), [PA-S|PP]) :- process(river(A),PA), process2(river(AA), PP).

process(lake(A), P) :- process(A,L), process(lake(L), P).
process(lake([]), []).
process(lake([A|AA]), [A|PP]) :- lake(A), !, process(lake(AA), PP).
process(lake([A|AA]), PP) :- process(lake(AA), PP).
process2(lake(A), P) :- process2(A,L), process2(lake(L), P).
process2(lake([]), []).
process2(lake([A-S|AA]), [PA-S|PP]) :- process(lake(A),PA), process2(lake(AA), PP).

process(state(A), P) :- process(A,L), process(state(L), P).
process(state([]), []).
process(state([A|AA]), [A|PP]) :- state(A), !, process(state(AA), PP).
process(state([A|AA]), PP) :- process(state(AA), PP).
process2(state(A), P) :- process2(A,L), process2(state(L), P).
process2(state([]), []).
process2(state([A-S|AA]), [PA-S|PP]) :- process(state(A),PA), process2(state(AA), PP).

process(mountain(A), P) :- process(A,L), process(mountain(L), P).
process(mountain([]), []).
process(mountain([A|AA]), [A|PP]) :- place(A), !, process(mountain(AA), PP).
process(mountain([A|AA]), PP) :- process(mountain(AA), PP).
process2(mountain(A), P) :- process2(A,L), process2(mountain(L), P).
process2(mountain([]), []).
process2(mountain([A-S|AA]), [PA-S|PP]) :- process(mountain(A),PA), process2(mountain(AA), PP).

% find the required (one-to-one); process2 generates pairwise list
process(len(A), P) :- process(A,L), process(len(L), P).
process(len([]), []).
process(len([A|AA]), [P|PP]) :- len(A, P), process(len(AA), PP).
process(len([A|AA]), PP) :- process(len(AA), PP).
process2(len(A), P) :- process(A,L), process2(len(L), P).
process2(len([]), []).
process2(len([A|AA]), [P-A|PP]) :- len(A, P), process2(len(AA), PP).
process2(len([A|AA]), PP) :- process2(len(AA), PP).

process(size(A), P) :- process(A,L), process(size(L), P).
process(size([]), []).
process(size([A|AA]), [P|PP]) :- size(A, P), process(size(AA), PP).
process(size([A|AA]), PP) :- process(size(AA), PP).
process2(size(A), P) :- process(A,L), process2(size(L), P).
process2(size([]), []).
process2(size([A|AA]), [P-A|PP]) :- size(A, P), process2(size(AA), PP).
process2(size([A|AA]), PP) :- process2(size(AA), PP).

process(area_1(A), P) :- process(A,L), process(area_1(L), P).
process(area_1([]), []).
process(area_1([A|AA]), [P|PP]) :- area(A, P), process(area_1(AA), PP).
process(area_1([A|AA]), PP) :- process(area_1(AA), PP).
process2(area_1(A), P) :- process(A,L), process2(area_1(L), P).
process2(area_1([]), []).
process2(area_1([A|AA]), [P-A|PP]) :- area(A, P), process2(area_1(AA), PP).
process2(area_1([A|AA]), PP) :- process2(area_1(AA), PP).

process(population_1(A), P) :- process(A,L), process(population_1(L), P).
process(population_1([]), []).
process(population_1([A|AA]), [P|PP]) :- population(A, P), process(population_1(AA), PP).
process(population_1([A|AA]), PP) :- process(population_1(AA), PP). % if not found
process2(population_1(A), P) :- process(A,L), process2(population_1(L), P).
process2(population_1([]), []).
process2(population_1([A|AA]), [P-A|PP]) :- population(A, P), process2(population_1(AA), PP).
process2(population_1([A|AA]), PP) :- process2(population_1(AA), PP). % if not found

process(density_1(A), P) :- process(A,L), process(density_1(L), P).
process(density_1([]), []).
process(density_1([A|AA]), [P|PP]) :- density(A, P), process(density_1(AA), PP).
process(density_1([A|AA]), PP) :- process(density_1(AA), PP).
process2(density_1(A), P) :- process(A,L), process2(density_1(L), P).
process2(density_1([]), []).
process2(density_1([A|AA]), [P-A|PP]) :- density(A, P), process2(density_1(AA), PP).
process2(density_1([A|AA]), PP) :- process2(density_1(AA), PP).

process(elevation_1(A), P) :- process(A,L), process(elevation_1(L), P).
process(elevation_1([]), []).
process(elevation_1([A|AA]), [P|PP]) :- elevation(A, P), process(elevation_1(AA), PP).
process(elevation_1([A|AA]), PP) :- process(elevation_1(AA), PP).
process2(elevation_1(A), P) :- process(A,L), process2(elevation_1(L), P).
process2(elevation_1([]), []).
process2(elevation_1([A|AA]), [P-A|PP]) :- elevation(A, P), process2(elevation_1(AA), PP).
process2(elevation_1([A|AA]), PP) :- process2(elevation_1(AA), PP).

%%%% no need for process2

process(capital_1(A), P) :- process(A,L), process(capital_1(L), P).
process(capital_1([]), []).
process(capital_1([A|AA]), [P|PP]) :- capital(A, P), process(capital_1(AA), PP).
process(capital_1([A|AA]), PP) :- process(capital_1(AA), PP).

% find all the required (one-to-many)
process(capital_2(A), P) :- process(A,L), process(capital_2(L), P).
process(capital_2([]), []).
process(capital_2([A|L]), P) :- findall(B, capital(B, A), AA),
				   process(capital_2(L),LL), append(AA,LL,P).

process(elevation_2(A), P) :- process(A,L), process(elevation_2(L), P).
process(elevation_2([]), []).
process(elevation_2([A|L]), P) :- findall(B, elevation(B, A), AA),
				   process(elevation_2(L),LL), append(AA,LL,P).

process(high_point_1(A), P) :- process(A,L), process(high_point_1(L), P).
process(high_point_1([]), []).
process(high_point_1([A|L]), P) :- findall(B, high_point(A, B), AA),
				   process(high_point_1(L),LL), append(AA,LL,P).
process2(high_point_1(A), P) :- process(A,L), process2(high_point_1(L), P).
process2(high_point_1([]), []).
process2(high_point_1([A|L]), [AA-A|P]) :- findall(B, high_point(A, B), AA),
				   process2(high_point_1(L),P).

process(higher_1(A), P) :- process(A,L), process(higher_1(L), P).
process(higher_1([]), []).
process(higher_1([A|L]), P) :- findall(B, higher(A, B), AA),
				   process(higher_1(L),LL), append(AA,LL,P).
process2(higher_1(A), P) :- process(A,L), process2(higher_1(L), P).
process2(higher_1([]), []).
process2(higher_1([A|L]), [AA-A|P]) :- findall(B, higher(A, B), AA),
				   process2(higher_1(L),P).

process(lower_1(A), P) :- process(A,L), process(lower_1(L), P).
process(lower_1([]), []).
process(lower_1([A|L]), P) :- findall(B, lower(A, B), AA),
				   process(lower_1(L),LL), append(AA,LL,P).
process2(lower_1(A), P) :- process(A,L), process2(lower_1(L), P).
process2(lower_1([]), []).
process2(lower_1([A|L]), [AA-A|P]) :- findall(B, lower(A, B), AA),
				   process2(lower_1(L),P).

process(loc_1(A), P) :- process(A,L), process(loc_1(L), P).
process(loc_1([]), []).
process(loc_1([A|L]), P) :- findall(B, loc(A, B), AA),
				   process(loc_1(L),LL), append(AA,LL,P).
process2(loc_1(A), P) :- process(A,L), process2(loc_1(L), P).
process2(loc_1([]), []).
process2(loc_1([A|L]), [AA-A|P]) :- findall(B, loc(A, B), AA),
				   process2(loc_1(L),P).

process(low_point_1(A), P) :- process(A,L), process(low_point_1(L), P).
process(low_point_1([]), []).
process(low_point_1([A|L]), P) :- findall(B, low_point(A, B), AA),
				   process(low_point_1(L),LL), append(AA,LL,P).
process2(low_point_1(A), P) :- process(A,L), process2(low_point_1(L), P).
process2(low_point_1([]), []).
process2(low_point_1([A|L]), [AA-A|P]) :- findall(B, low_point(A, B), AA),
				   process2(low_point_1(L),P).

process(next_to_1(A), P) :- process(A,L), process(next_to_1(L), P).
process(next_to_1([]), []).
process(next_to_1([A|L]), P) :- findall(B, next_to(A, B), AA),
				   process(next_to_1(L),LL), append(AA,LL,P).
process2(next_to_1(A), P) :- process(A,L), process2(next_to_1(L), P).
process2(next_to_1([]), []).
process2(next_to_1([A|L]), [AA-A|P]) :- findall(B, next_to(A, B), AA),
				   process2(next_to_1(L),P).

process(traverse_1(A), P) :- process(A,L), process(traverse_1(L), P).
process(traverse_1([]), []).
process(traverse_1([A|L]), P) :- findall(B, traverse(A, B), AA),
				   process(traverse_1(L),LL), append(AA,LL,P).
process2(traverse_1(A), P) :- process(A,L), process2(traverse_1(L), P).
process2(traverse_1([]), []).
process2(traverse_1([A|L]), [AA-A|P]) :- findall(B, traverse(A, B), AA),
				   process2(traverse_1(L),P).

process(high_point_2(A), P) :- process(A,L), process(high_point_2(L), P).
process(high_point_2([]), []).
process(high_point_2([A|L]), P) :- findall(B, high_point(B, A), AA),
				   process(high_point_2(L),LL), append(AA,LL,P).
process2(high_point_2(A), P) :- process(A,L), process2(high_point_2(L), P).
process2(high_point_2([]), []).
process2(high_point_2([A|L]), [AA-A|P]) :- findall(B, high_point(B, A), AA),
				   process2(high_point_2(L),P).

process(higher_2(A), P) :- process(A,L), process(higher_2(L), P).
process(higher_2([]), []).
process(higher_2([A|L]), P) :- findall(B, higher(B, A), AA),
				   process(higher_2(L),LL), append(AA,LL,P).
process2(higher_2(A), P) :- process(A,L), process2(higher_2(L), P).
process2(higher_2([]), []).
process2(higher_2([A|L]), [AA-A|P]) :- findall(B, higher(B, A), AA),
				   process2(higher_2(L),P).

process(lower_2(A), P) :- process(A,L), process(lower_2(L), P).
process(lower_2([]), []).
process(lower_2([A|L]), P) :- findall(B, lower(B, A), AA),
				   process(lower_2(L),LL), append(AA,LL,P).
process2(lower_2(A), P) :- process(A,L), process2(lower_2(L), P).
process2(lower_2([]), []).
process2(lower_2([A|L]), [AA-A|P]) :- findall(B, lower(B, A), AA),
				   process2(lower_2(L),P).

process(loc_2(A), P) :- process(A,L), process(loc_2(L), P).
process(loc_2([]), []).
process(loc_2([A|L]), P) :- findall(B, loc(B, A), AA),
				   process(loc_2(L),LL), append(AA,LL,P).
process2(loc_2(A), P) :- process(A,L), process2(loc_2(L), P).
process2(loc_2([]), []).
process2(loc_2([A|L]), [AA-A|P]) :- findall(B, loc(B, A), AA),
				   process2(loc_2(L),P).

process(low_point_2(A), P) :- process(A,L), process(low_point_2(L), P).
process(low_point_2([]), []).
process(low_point_2([A|L]), P) :- findall(B, low_point(B, A), AA),
				   process(low_point_2(L),LL), append(AA,LL,P).
process2(low_point_2(A), P) :- process(A,L), process2(low_point_2(L), P).
process2(low_point_2([]), []).
process2(low_point_2([A|L]), [AA-A|P]) :- findall(B, low_point(B, A), AA),
				   process2(low_point_2(L),P).

process(traverse_2(A), P) :- process(A,L), process(traverse_2(L), P).
process(traverse_2([]), []).
process(traverse_2([A|L]), P) :- findall(B, traverse(B, A), AA),
				   process(traverse_2(L),LL), append(AA,LL,P).
process2(traverse_2(A), P) :- process(A,L), process2(traverse_2(L), P).
process2(traverse_2([]), []).
process2(traverse_2([A|L]), [AA-A|P]) :- findall(B, traverse(B, A), AA),
				   process2(traverse_2(L),P).

process(next_to_2(A), P) :- process(A,L), process(next_to_2(L), P).
process(next_to_2([]), []).
process(next_to_2([A|L]), P) :- findall(B, next_to(B, A), AA),
				   process(next_to_2(L),LL), append(AA,LL,P).
process2(next_to_2(A), P) :- process(A,L), process2(next_to_2(L), P).
process2(next_to_2([]), []).
process2(next_to_2([A|L]), [AA-A|P]) :- findall(B, next_to(B, A), AA),
				   process2(next_to_2(L),P).

process(longer(A), P) :- process(A,L), process(longer(L), P).
process(longer([]), []).
process(longer([A|L]), P) :- findall(B, longer(B, A), AA),
				   process(longer(L),LL), append(AA,LL,P).
process2(longer(A), P) :- process(A,L), process2(longer(L), P).
process2(longer([]), []).
process2(longer([A|L]), [AA-A|P]) :- findall(B, longer(B, A), AA),
				   process2(longer(L),P).
% metas
  %  helpful for meta
pair_size([A|AA], [(Size-A)|LL]) :- size(A,Size), pair_size(AA, LL).
pair_size([A|AA], LL) :- pair_size(AA, LL).
pair_size([], []).
pair_elevation([A|AA], [(Elevation-A)|LL]) :- elevation(A,Elevation), pair_elevation(AA,LL).
pair_elevation([A|AA], LL) :- pair_elevation(AA,LL).
pair_elevation([], []).
pair_len([A|AA], [(Len-A)|LL]) :- len(A,Len), pair_len(AA, LL).
pair_len([A|AA], LL) :- pair_len(AA, LL).
pair_len([], []).

process(largest(A), PP) :-  process(A,P), pair_size(P, PS),
			      (PS=[] -> PP=[]; (max_key(PS, M),PP=[M])).
process(smallest(A), PP) :-  process(A,P), pair_size(P, PS),
			       (PS=[] -> PP=[]; (min_key(PS, M),PP=[M])).

process(highest(A), PP) :-  process(A,P), pair_elevation(P, PS),
			      (PS=[] -> PP=[]; (max_key(PS, M),PP=[M])).
process(lowest(A), PP) :-  process(A,P), pair_elevation(P, PS),
			     (PS=[] -> PP=[]; (min_key(PS, M),PP=[M])).

process(longest(A), PP) :-  process(A,P), pair_len(P, PS),
			    (PS=[] -> PP=[]; (max_key(PS, M),PP=[M])).
process(shortest(A), PP) :- process(A,P), pair_len(P, PS),
			      (PS=[] -> PP=[]; (min_key(PS, M),PP=[M])).

% ones
numerify([],[]).
numerify([L-S|R], [N-S|NR]) :- sort(L,LL), length(LL,N), numerify(R,NR).

process(largest_one(A), P) :-  process2(A, S),
				 (S=[]-> P=[]; (max_key(S,M), P=[M])).
process(highest_one(A), P) :-  process2(A, S),
				 (S=[]-> P=[]; (max_key(S,M), P=[M])).
process(longest_one(A), P) :-  process2(A, S),
				 (S=[]-> P=[]; (max_key(S,M), P=[M])).
process(most(A), P) :- process2(A, S),numerify(S,NS),
			 (S=[]-> P=[]; (max_key(NS,M), P=[M])).

process(smallest_one(A), P) :-  process2(A, S),
				(S=[]-> P=[]; (min_key(S,M), P=[M])).
process(lowest_one(A), P) :-  process2(A, S),
			      (S=[]-> P=[]; (min_key(S,M), P=[M])).
process(shortest_one(A), P) :-  process2(A, S),
				(S=[]-> P=[]; (min_key(S,M), P=[M])).
process(fewest(A), P) :- process2(A, S),numerify(S,NS),
			 (S=[]-> P=[]; (min_key(NS,M), P=[M])).


process(count(A), [P]) :- process(A, B), sort(B, BB), length(BB, P).
process(sum(A), [P]) :- process(A, B), sumlist(B, 0, P).

% what's the meaning of each really? -ywwong
process(each(Q), P) :- process(Q, P).

% exclude and intersection
 % helpful: remove all occurrences of elements of the second list from the first list
minus(L,[],L).
minus(L, [A|AA], P) :- delete(L,A,L2), minus(L2, AA, P).
 % helpful: intersection of two lists
intersect([],L,[]).
intersect([A|L1], L2, [A|L]) :- member(A,L2), intersect(L1, L2, L).
intersect([B|L1], L2, L) :- intersect(L1, L2, L).

process(exclude(A, B), P) :- process(A,P1), process(B,P2), minus(P1, P2, P).
process(intersection(A, B), P) :- process(A,P1), process(B,P2), intersect(P1, P2, P).
