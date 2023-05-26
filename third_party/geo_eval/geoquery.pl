:- ensure_loaded(library('lists')).
:- ensure_loaded(library('ordsets')).
%:- ensure_loaded(geobase).

country(countryid(usa)).

state(stateid(State)) :- state(State,_,_,_,_,_,_,_,_,_).

city(cityid(City,St)) :- city(_,St,City,_).

river(riverid(R)) :- river(R,_,_). 

place(placeid(P)) :- highlow(_,_,P,_,_,_).
place(placeid(P)) :- highlow(_,_,_,_,P,_).

mountain(A) :- mountain(A,_,_,_). % Rohit		     
lake(A) :- lake(A,_,_). % Rohit		     

abbreviation(stateid(State), Ab) :- 
	state(State,Ab,_,_,_,_,_,_,_,_).
abbreviation(Ab) :- abbreviation(_,Ab).

capital(stateid(State), cityid(Cap,St)) :- state(State,St,Cap,_,_,_,_,_,_,_).
capital(Cap) :- capital(_,Cap).

print_name(stateid(X),X) :- !.
print_name(cityid(X,_), X) :- !.
print_name(riverid(X), X) :- !.
print_name(placeid(X), X) :- !.
print_name(Goal, Y) :- (Goal=_/_;Goal=_*_;Goal=_+_;Goal=_-_),!, Y is Goal.
print_name(X,X).

loc(X,countryid(usa)) :-
	city(X) ; state(X) ; river(X) ; place(X).
loc(cityid(City,St), stateid(State)) :-
	city(State, St, City,_).
loc(placeid(P), stateid(S)) :- highlow(S,_,P,_,_,_). % Rohit
loc(placeid(P), stateid(S)) :- highlow(S,_,_,_,P,_). % Rohit
%loc(placeid(P), stateid(S)) :-
%	( highlow(S,_,P,_,_,_) ->
%	      true
%	; highlow(S,_,_,_,P,_)
%	).
loc(riverid(R), stateid(S)) :-
	river(R,_,States),
	member(S,States).

traverse(riverid(R), stateid(S)) :-
	river(R,_,States),
	member(S,States).

high_point(countryid(usa), placeid('mount mckinley')).
high_point(stateid(S), placeid(P)) :-
	 highlow(S,_,P,_,_,_).

low_point(countryid(usa), placeid('death valley')).
low_point(stateid(S), placeid(P)) :-
	 highlow(S,_,_,_,P,_).

area(stateid(X),Areal) :-
	state(X,_,_,_,Area,_,_,_,_,_),
	Areal is float(Area).

major(cityid(C,S)) :-
	X = cityid(C,S),
	city(X),
	population(X,P),
	P > 150000.
major(riverid(R)) :-
	X = riverid(R),
	river(X),
	len(X,L),
	L > 750.
	
first(G) :- (G -> true).

n_solutions(N,Goal) :-
	findall(Goal, Goal, GList0),
	length(Solutions, N),
	append(Solutions,_,GList0),
	member(Goal, Solutions).

nth_solution(N,Goal) :-
	findall(Goal, Goal, GList),
	nth(N,GList,Goal).

population(stateid(X),Pop) :- 
	state(X,_,_,Pop,_,_,_,_,_,_).
population(cityid(X,St), Pop) :-
	city(_,St,X,Pop).

len(riverid(R), L) :-
	river(R,L,_).

elevation(placeid(P),E) :- highlow(_,_,_,_,P,E). % Rohit
elevation(placeid(P),E) :- highlow(_,_,P,E,_,_). % Rohit
%elevation(placeid(P),E) :-
%	( highlow(_,_,P,E,_,_) 
%	; highlow(_,_,_,_,P,E)
%	).

size(stateid(X), S) :-
	area(stateid(X), S).
size(cityid(X,St), S) :-
	population(cityid(X,St), S).
size(riverid(X), S) :-
	len(riverid(X),S).
size(placeid(X), S) :-
	elevation(placeid(X),S).
size(X,X) :-
	number(X).
	
next_to(stateid(X),stateid(Y)) :-
	border(X,_,Ys),
	member(Y,Ys).

density(S,D) :-
	population(S,P),
	area(S,A),
	D is P / A.

largest(Var, Goal) :-
	findall(Size-Goal, (Goal,size(Var,Size)), Pairs0),
	max_key(Pairs0, Goal).

max_key([Key-Value|Rest],Result) :-
	max_key(Rest, Key, Value, Result).

max_key([], _, Value, Value).
max_key([K-V|T], Key, Value, Result):-
	( K > Key ->
	     max_key(T, K, V, Result)
	; max_key(T, Key, Value, Result)
	).

smallest(Var, Goal) :-
	findall(Size-Goal, (Goal,size(Var,Size)), Pairs0),
	min_key(Pairs0, Goal).

min_key([Key-Value|Rest],Result) :-
	min_key(Rest, Key, Value, Result).

min_key([], _, Value, Value).
min_key([K-V|T], Key, Value, Result):-
	( K < Key ->
	     min_key(T, K, V, Result)
	; min_key(T, Key, Value, Result)
	).

count(V,Goal,N) :-
	findall(V,Goal,Ts),
	sort(Ts, Unique),
	length(Unique, N).

at_least(Min,V,Goal) :-
	count(V,N,Goal),
	Goal,  % This is a hack to instantiate N, making this order independent.
	N >= Min.

at_most(Max,V,Goal) :-
	count(V,Goal,N),
	N =< Max.

execute_query(Query, Unique):-
	tq(Query, answer(Var,Goal)),
	findall(Name, (Goal, print_name(Var,Name)), Answers),
	sort(Answers, Unique).
%---------------------------------------------------------------------------
tq(G,G) :-
	var(G), !.
tq(largest(V,Goal), largest(Vars, DVars, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,V,Goal),(DVars,DV,Goal1)),
	tq(Goal1,DGoal).
tq(smallest(V,Goal), smallest(Vars, DVars, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,V,Goal),(DVars,DV,Goal1)),
	tq(Goal1,DGoal).
tq(highest(V,Goal), highest(Vars, DVars, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,V,Goal),(DVars,DV,Goal1)),
	tq(Goal1,DGoal).
tq(most(I,V,Goal), most(Vars, DVars, DI, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,I,V,Goal),(DVars,DI,DV,Goal1)),
	tq(Goal1,DGoal).
tq(fewest(I,V,Goal), fewest(Vars, DVars, DI, DV, DGoal)) :-
	!,
	variables_in(Goal, Vars),
	copy_term((Vars,I,V,Goal),(DVars,DI,DV,Goal1)),
	tq(Goal1,DGoal).
tq(Goal,TGoal) :-
	functor(Goal,F,N),
	functor(TGoal,F,N),
	tq_args(N,Goal,TGoal).

tq_args(N,Goal,TGoal) :-
	( N =:= 0 ->
	     true
	; arg(N,Goal,GArg),
	  arg(N,TGoal,TArg),
	  tq(GArg,TArg),
	  N1 is N - 1,
	  tq_args(N1,Goal,TGoal)
	).

variables_in(A, Vs) :- variables_in(A, [], Vs).
	
variables_in(A, V0, V) :-
	var(A), !, add_var(V0, A, V).
variables_in(A, V0, V) :-
	ground(A), !, V = V0. 
variables_in(Term, V0, V) :-
	functor(Term, _, N),
	variables_in_args(N, Term, V0, V).

variables_in_args(N, Term, V0, V) :-
	( N =:= 0 ->
	      V = V0
	; arg(N, Term, Arg),
	  variables_in(Arg, V0, V1),
	  N1 is N-1,
	  variables_in_args(N1, Term, V1, V)
	).

add_var(Vs0, V, Vs) :-
	( contains_var(V, Vs0) ->
	      Vs = Vs0
	; Vs = [V|Vs0]
	).


contains_var(Variable, Term) :-
	\+ free_of_var(Variable, Term).

%   free_of_var(+Variable, +Term)
%   is true when the given Term contains no sub-term identical to the
%   given Variable (which may actually be any term, not just a var).
%   For variables, this is precisely the "occurs check" which is
%   needed for sound unification.

free_of_var(Variable, Term) :-
	Term == Variable,
	!,
	fail.
free_of_var(Variable, Term) :-
	compound(Term),
	!,
	functor(Term, _, Arity),
	free_of_var(Arity, Term, Variable).
free_of_var(_, _).

free_of_var(1, Term, Variable) :- !,
	arg(1, Term, Argument),
	free_of_var(Variable, Argument).
free_of_var(N, Term, Variable) :-
	arg(N, Term, Argument),
	free_of_var(Variable, Argument),
	M is N-1, !,
	free_of_var(M, Term, Variable).

%---------------------------------------------------------------------------
/*
execute_query(answer(Var, Goal), Unique) :-
	findall(Name,(Goal,print_name(Var,Name)),Answers),
	sort(Answers,Unique).
*/
answer(Var, Goal) :- 
	nl,nl,
	findall(Name,(Goal,print_name(Var,Name)),Answers),
	sort(Answers,Unique),
	format('Answer = ~w~n',[Unique]).

sum(V, Goal, X) :-
	findall(V, Goal, Vs),
	sumlist(Vs, 0, X).

highest(Vars, DVars, DV, Goal) :-
	highest(DV, Goal), !,
	Vars = DVars.

highest(X, Goal) :-
	largest(Y, (Goal, elevation(X,Y))).
/*CAT. bug
lowest(X,Goal) :-
	largest(Y, (Goal, elevation(X,Y))).
*/
lowest(X,Goal) :-
	smallest(Y, (Goal, elevation(X,Y))).

shortest(X,Goal) :-
	smallest(Y, (Goal, len(X,Y))).

longest(X,Goal) :-
	largest(Y, (Goal, len(X,Y))).


higher(X,Y) :-
	elevation(X,EX),
	elevation(Y,EY),
	EX > EY.

%---------------------------------
%CAT added
lower(X, Y) :-
	elevation(X,EX),
	elevation(Y,EY),
	EX < EY.

longer(X, Y) :-
	len(X,LX),
	len(Y, LY),
	LX > LY.

shorter(X, Y) :-
	len(X,LX),
	len(Y, LY),
	LX < LY.

more(X, Y) :-
	X > Y.
%---------------------------------

divide(X,Y, X/Y).
multiply(X,Y,X*Y).
add(X,Y,X+Y).
%subtract(X,Y,X-Y).

sumlist([], Sum, Sum).
sumlist([V|Vs], Sum0, Sum) :-
	Sum1 is Sum0 + V,
	sumlist(Vs, Sum1, Sum).

const(V, V).

largest(Vars, DVars, DV, DGoal) :-
	largest(DV, DGoal),!,
	Vars = DVars.

smallest(Vars, DVars, DV, DGoal) :-
	smallest(DV, DGoal),!,
	Vars = DVars.

most(Vars, DVars, DI, DV, DGoal) :-
	most(DI, DV, DGoal),!,
	Vars = DVars.

fewest(Vars, DVars, DI, DV, DGoal) :-
	fewest(DI, DV, DGoal),!,
	Vars = DVars.

most(Index,Var,Goal) :-
	setof(Index-Var, Goal, Solutions),
	keysort(Solutions, Collect),
	maximum_run(Collect, Index).

maximum_run(Solutions, Index) :-
	maximum_run(Solutions, foo, 0, Index).

maximum_run([], Index, _Count, Index) :- !.
maximum_run([Index1-_|Rest], BestIndex0, Count0, BestIndex) :-
	first_run(Rest, Index1, 1, Count1, Rest1),
	( Count1 > Count0 ->
	     BestIndex2 = Index1,
	     Count2 = Count1
	; BestIndex2 = BestIndex0,
	  Count2 = Count0
	),
	maximum_run(Rest1, BestIndex2, Count2, BestIndex).

first_run([], _Index, N, N, []).
first_run([Index-G|Rest0], Target, N0, N, Rest) :-
	( Target = Index ->
	     N1 is N0 + 1,
	     first_run(Rest0, Target, N1, N, Rest)
	; N = N0,
	  Rest = [Index-G|Rest0]
	).

fewest(Index,Var,Goal) :-
	setof(Index-Var, Goal, Solutions),
	keysort(Solutions, Collect),
	minimum_run(Collect, Index).

minimum_run(Solutions, Index) :-
	minimum_run(Solutions, foo, 1000, Index).

minimum_run([], Index, _Count, Index) :- !.
minimum_run([Index1-_|Rest], BestIndex0, Count0, BestIndex) :-
	first_run(Rest, Index1, 1, Count1, Rest1),
	( Count1 < Count0 ->
	     BestIndex2 = Index1,
	     Count2 = Count1
	; BestIndex2 = BestIndex0,
	  Count2 = Count0
	),
	minimum_run(Rest1, BestIndex2, Count2, BestIndex).


