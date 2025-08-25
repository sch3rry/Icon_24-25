% --- Knowledge Base Colture Agricole ---
% Ogni fatto crop/10: 
% crop(Nome, N, P, K, temp_MIN, temp_MAX, ph_%, humidity, piogge, costo_per_acro)

crop(rice, 80 , 47, 40 , 20 , 27 , 6.4 , 82 , 236 , 1000).
crop(maize, 78 , 48 , 20 , 18 , 26 , 6.2 , 65 , 84 , 700).
crop(chickpea, 40 , 68 , 80 , 17 , 21, 7.3 , 16 , 80 , 900).
crop(kidneybeans, 21 , 67 , 20 , 15 , 25 , 5.7 , 95 , 176 , 600).
crop(pigeonpeas, 21 , 68 , 20 , 18 , 37 , 5.7 , 48 , 149 , 400).
crop(mothbeans, 21 , 48 , 20 , 24 , 31 , 6.8 , 53 , 51 , 400).
crop(mungbean, 21 , 47 , 20 , 27 , 30 , 6.7 , 85 , 48 , 400).
crop(blackgram, 40 , 67 , 19 , 25 , 35 , 7.1 , 65 , 68 , 400).
crop(lentil, 19 , 68 , 19 , 18 , 30 , 6.9 , 65 , 46 , 500).
crop(pomegranate, 19 , 19 , 40 , 18 , 25 , 6.4 , 90 , 107 , 2500).
crop(banana, 100 , 82 , 50 , 25 , 30 , 5.9 , 80 , 105 , 3000).
crop(mango, 20 , 27 , 30 , 27 , 36 , 5.7 , 50 , 95 , 2000).
crop(grapes, 23 , 132 , 200 , 8 , 12 , 6.0 , 82 , 69 , 5000).
crop(watermelon, 99 , 17 , 50 , 24 , 27 , 6.5 , 85 , 51 , 2000).
crop(muskmelon, 100 , 18 , 50 , 27 , 27 , 6.3 , 92 , 24.7 , 2000).
crop(apple, 20 , 134 , 200 , 21 , 24 , 5.9 , 92 , 113 , 10000).
crop(orange, 19 , 16 , 10 , 10 , 35 , 7.0 , 92 , 110 , 3000).
crop(papaya, 50 , 59 , 50 , 23 , 44 , 6.7 , 92 , 143 , 1500).
crop(coconut, 22 , 17 , 31 , 25 , 30 , 5.9 , 95 , 176 , 1500).
crop(cotton, 118 , 46 , 20 , 22 , 26 , 6.9 , 80 , 80 , 1500).
crop(jute, 78 , 47 , 40 , 23 , 27 , 6.7 , 80 , 175 , 1000).
crop(coffee, 101 , 29 , 30 , 23 , 28 , 6.8 , 58 , 158 , 4000).

% month(NomeMese, TempMin, TempMax)
month(january, 2, 12).
month(february, 3, 13).
month(march, 6, 16).
month(april, 10, 20).
month(may, 15, 25).
month(june, 18, 28).
month(july, 20, 30).
month(august, 19, 29).
month(september, 15, 25).
month(october, 10, 20).
month(november, 5, 15).
month(december, 2, 10).

% --- Regole per filtrare le colture ---
% Regola: un crop è adatto a un mese se il range di temperatura del mese è compreso in quello ottimale della coltura
crop_suitable_for_month(Crop, Month) :-
	crop(Crop,_, _, _, Tmin, Tmax, _, _, _, _),
	month(Month, TminMonth, TmaxMonth),
	Tmin =< TminMonth,
	Tmax >= TmaxMonth.

% Regola: suggerire colture in base a un mese
suggest_crops_by_month(Month, Crops) :-
    findall(Crop, crop_suitable_for_month(Crop, Month), Crops).


% Colture seminabili con budget e terreno disponibili
sowing_economical_info(Acri, Budget, Crop) :-
    crop(Crop, _, _, _, _, _, _, _, _, Costo),
    Totale is Acri * Costo,
    Totale =< Budget.

% Coltura con costo minimo tra quelle possibili
min_cost_crop(Acri, Budget, BestCrop) :-
    findall((CostoTot, C), 
        (sowing_economical_info(Acri, Budget, C),
		crop(C, _, _, _, _, _, _, _, _, Costo),
		CostoTot is Acri * Costo),
        Lista),
        Lista \= [],
    sort(Lista, [(_, BestCrop)|_]).

% Colture per condizioni climatiche
sowing_zone_info(PHzone, UmidZone, RainZone, Crops) :-
    findall(Crop,
        (
            crop(Crop, _, _, _, _, _, PH, Umid, Rain, _),
            ph_category(PH, PHzone),
            humidity_category(Umid, UmidZone),
            rainfall_category(Rain, RainZone)
        ),
        Crops).

% --- Classificazione dei valori climatici ---
ph_category(PH, acido)  :- PH < 6.5.
ph_category(PH, neutro) :- PH >= 6.5, PH =< 7.5.
ph_category(PH, basico) :- PH > 7.5.

humidity_category(Val, bassa) :- Val < 60.
humidity_category(Val, media) :- Val >= 60, Val =< 75.
humidity_category(Val, alta)  :- Val > 75.

rainfall_category(Rain, arida)    :- Rain < 75.
rainfall_category(Rain, standard) :- Rain >= 75, Rain =< 150.
rainfall_category(Rain, piovosa)  :- Rain > 150.

% --- Coltura più simile ad un’altra ---
% Usa distanza euclidea sulle feature numeriche
most_sim(RefCrop, SimCrop) :-
    crop(RefCrop, N1, P1, K1, Tmin1, Tmax1, PH1, H1, R1, _),
    findall((Dist, C), 
        (crop(C, N2, P2, K2, Tmin2, Tmax2, PH2, H2, R2, _),
         C \= RefCrop,
         Dist is sqrt((N1-N2)^2 + (P1-P2)^2 + (K1-K2)^2 + (Tmin1-Tmin2)^2 + (Tmax1-Tmax2)^2 + (PH1-PH2)^2 +  (H1-H2)^2 + (R1-R2)^2)),
        Lista),
    sort(Lista, [(_, SimCrop)|_]).

% --- Coltura consigliata combinando condizioni climatiche e budget ---
recommended_crop(Acri, Budget, PHzone, UmidZone, RainZone, Crop) :-
    sowing_zone_info(PHzone, UmidZone, RainZone, Crops),
% ---   member(Crop, Crops) ---
    crop(Crop, _, _, _, _, _, _, _, _, Costo),
    Totale is Acri * Costo,
    Totale =< Budget.
