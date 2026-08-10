"""Initial immutable schema snapshot.

Revision ID: 20260806_0001
Revises:
Create Date: 2026-08-06

The payload below is a compressed *literal* copy of the DDL that existed when
this revision was introduced.  It deliberately has no dependency on runtime
modules: changing pipeline code must never rewrite migration history.
"""

from __future__ import annotations

import base64
import gzip
from typing import Sequence, Union

from alembic import op

revision: str = "20260806_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DDL_BOUNDARY = "\n\n-- GRIDIRON_ORACLE_0001_DDL_BOUNDARY\n\n"
_INITIAL_SCHEMA_DDL_GZIP_B85 = (
    "ABzY8000000u${VU325M@!h`ybDi|0iG8-Omt5;i9%Nfi)c9nd>`QWe;XouLF{Vf!KV&Npo&Jc<^au5y^p|uO06~BRDEdZi`cS{ziC8QayNmt8k_Q*F@#ua`?nmd>V=}oS)0=xT{&{kDe@6;N*Ety+5cmrNqW*py%`Uzl%?76@Lqcw6lXs)pNAkn?;~|HP>6%rL_v4@M#Zx~lw(X~RL4&YdH=brmvha#+%1qcYTTq`hV9PY11q(b{z|+aQ@!kFC-R=Ef2;ax_!}T?}9AAw-T*Hr>p9aH&;Wq~d9a~N#x@DPd(|k?!qqXZr8AV>T^_sji$-^Q{;s({oS<+_5iiOKn(R%!-Kfb!Y8Oi7KFe_Fb?jL?!!j>eflPF?K*2J=9G|N5ql;V%c^nU!^cqT?zTx44h$D1P{pQ6kybkZPSC52aJk@lG7Y3x-z(|EAR)KiD#_940@FyAV;-NMTm&83XQpJamgQIS&#gFP6iJ7AHdN|Bo^OmIk31A{^n4Sl5BzOWo`?k5>D$<r`7N0HosZtJbpI)y_;SRZUdJK2LV%YM8qXS^k$_Car*n{{f;(oVx;7zeJgJk3C5y(o-XYph5L8fioN80XhIiP=_;Yh#?#U1J%`3*wZ5bCl2hG*`w%+diMz9O`o}yeL^Z&xZbYH=a#K*S0evETd<itrSun-P!nRJR46h#>%D+!eFQ!3l|xksQ4ML5w*^(^<3^azdh!Nb&Qj2*G|o<yTOt)1+9q-&j#;s9d0l`S*H<0z}&eXsXYZApCa$txm=p&c^EIfEzN=!xTfY6ToL3wF@|x$GM}Xey;evXX6^MYYjYTnE&#9R)52dl;1*$)7haG&>U^@U%u?FkjRm8hCaeq2r&$)Vob2XzR_3cd{I1}3^Sgpvl<Rraq``z_J_|PvSTh)){(~K%YxP~6#I;k0>Y5k}nk`w;i<fFY7H&4y7H}4BIxt#v(u&cdlU6v3PC8b$a8k!e{mf~2bXUi@=;bS#b!x8Krwvqhxk=h<-DI&~aSm}wOw(>&SCFY+bYfi4xS;vgOB1jtd0#j$HR#pgE^elG_p=euLds8`a+#j=dia2Rm`>h*7?XkWgol+U<FTo9!ISCb_~(WT4uhvQPT^Ge>Niu}OKbQA%yW=SSA|~Af?A9oawoVWYzW|Ft~?*)UsREU9s+PTZS#_*BX#>;J47qWHHUSdSi3+$YDVJ+DbvO*EtauxS8}6=2z8bKLz9Oxt$1v5%e6)~$7A+XELrSf()C|tIo&KlYf_p~!CBo7z>tr&T&$sutm%`Nr;G({+ktabOybdbvSFBYGQSWDXt4VdjKR5vWEx!kJkg+9ibJLgXNR)xLpT^~cv|4ra<k4sk=EdLbBjCWgS9Jj$pm+35l0uR7Ddcgl|0g8ecOQs$ZFWm0SgzM%o77)?gd#Q^2GC-o9pptswxRrqwBk|0`n6HM{_Jf%xM(u0!BwpA^0(X>tV<+5tE8(ObCOmk|~GP_WS5~#n*;<*X&kEpP|>)?8?RUGCDg%`h0<*wDQI=sAO<bWjU3kxELS+bpYytq65Tbb)MnO8+>!fL~-@=j1HLOBJ{(Qo6Y|LF=a^!GVro8{!g;jx$6d#JGk?C>;|Vb$3o7Y9Pc8@RK;7b*|OMQd1=P-+~n?t;eNm>|5C0~V@8(HKs@d5JB67(I65NV%_f(V+0E3unT;;4$KJ`w$*FgFdF`Fwe3)JWhJ$Vk0amTdn75`y7C!YUA$@)oX}^V}eb$mbvz+9f)>+)?UsUQ}S27^MU162cEx&i8>uaD~K!XyUm;M7Q$mtpRfktZzL0popLG2PS$yu0WWX<5=mt;e;kj6eEDa-if!H{Ek;FCNs;GQ)-*VI~qR)|{gZBP@dmdy9UD&Xx{two0ptW9_~t*xTR4qA2iE?OJ;`c7JFcso{WA)o_m6W)$hCFOcH-hs6VZ^vpaw{>7`!aJm8E6sIDs{!AQwQ!V@Tv&^CyslwwCJVc;HsKw7wH27V_}YNCW3?87JFqt49hzmUPIhTl1K!Qo=jcEaz8h;_9caR9ptXg5eMattBRi)#!;2~9+<0&#t4~2!4UPk28t*8j>i23DsG~k>19cRA-B?>iNE@itrnEtYaRTooarXkPQ@RcU9Myg+2khm3Hvx!yy9=x1ddCLpxZi1lYGs^eJ=Usxe>fxWFgue$I>aM3HS!>PP98x8^#ZX3m1?9wY_;SHG-!i4KsDVkrdU7?(P+Td{k}OPKeD)F>T23skUr&bys7rq>8b}P)-w7>YVzjXgNS{x8=6x*7hBm4&57;}T{G#cGjfmd=mlo*PleVO(Y$x>cXRIst=IhejLg8qisAy@=Eh}vm+1>JaDl|lbA<KgCINcB1;flE;AgPpEs$463|PVz3&>$1X3u3oFGBwbPf7pH{_%`V;$KQI5XWS+p<zVlVT9dQgGH9C$@uPeN@ze+@Sop^B<NcL=f7NoVy_sDiq%#`=^@ABiJl)1^3)_S&X&0y;Z4?Ip0fbvfRrA(H-8t?)Xdr?h4CUn8K+x<_d~G%68LdQHX$W3q!}~{e`T*yNhAjWYvd(%CRthZr!#WN@M0<4Fwn+0fX!9>L4SBO-yY$=l0UG17?Il*cu?{N6sVw}{TT~}%2fOciI)A2s%2HshdUsfpmPx0uU;nlxw=7l0Ri)J3$%*`cm9fjDVbm-MSW3=7ukJI2J9(~KrjXIvn0=t0A!sH<?RZ?U2wZ@^bQ%z@wJ-a|3Cjh2B+l6VA)XC^EN=llDQCTNR1!>FltN{^%2fBpg%!|xH!jA8OkG3)l8R;ueu5G0tU+j4a)iv*0FOfn?5-s<0v6BST4rfIj)cE=$PL&uqXkgH#8C+k$ej&_ZmzOS%l#5rB3o0uqmA7Ry091fWFM-MJt{MBmo=xfCexDMM8=dU{@E29Cg5+fjVG4Q6qCYh}Kb8wT&j_3Qq@x4m<{#qJJ?;ls*Pw+rBims%SqlC>VSRD8ChT8+n}#_IHCZ$dV1!LLtOfA(&3^t8ki=xnm&!J(8g4f(e3F27Uo<2bZbz>i+7U+~F;o53Ysh4F${3WZ8_Qt8I=2Mntg)!Tkt*t8a&oYfG$D^?t>G{6_{F4kl+4CS@LN321yEyaUMe<kblX7Z<+-OPRCc6|f=$zJeu>%5|I%hGZ~7r9A<ieHykqUAsjl$(J)=r7-3f00Z`F`HH-K1IvF)zJB@o#jxEh+Y9rf!kmKZ337QSK^Ksx<AYsjrdQVjoCxs2>Cs!|Y2g75RR<MWDq>u5S?U)h-qP6;_vXE-0Cl!|wLzWTS#3~jHtrz6)z#Dnb+#*6KwX_cc2H~MP8-zOHPi-m^)*>Qu@5Q-+z%|?1^Q|3t-`_rY$3p&z;<)Moxn9j(A7!TP?s)H)V@+RZCgsrS{t{$i`&+2W8N|BvFE0NxN%F@xY^LJiVjtQ{@IIIyL*~z*;;(|aa$upc<{1Cl5H(lS<QDinwM7kTMP}*j%$+L-3_FF+mC%+>xGE{k+BprH{+EB1tCoDi<`1|aY$ZVCgr?<z2g7hmyjC*V!YYM-`C+f1P#akUI8~a`1JOLAcDuuJ~I|Emv|QjM||TIEW{akxQuy7si#?HGjW5?A7aLd`JS#JwSrTLKSuEbCy^EYt1ACY%H~0=x(=kXnZfCKT!HbC&Q2t35Qs5;kld+Ul@|*<K3Hbvmj#AP;zbLIs4H>|d6p063Uohiq}u{5!sar%#YP>pMb#2yz^zBPARwP8B9qy%R6a=HWE{2O*Ii~?w#mwu0^`d=^0LlvvqJlw#LEPBfKLRaFQ3z!w3lO>&eO__<j+5})>}|J=x=p;fkorn5m_@(W*@dV=KWxxKDVNOa})$sgar8)uX*yOmS(givdqnrW8-&>kkJ{_jIypVt(qbs20zV?K^&jJ^+7Ccu8m&PzIKH~RAdLUb5fOXe2D{GeuW16TILWeG;q8>S7;Q(0FE6fB9^IO7((qJ^udIypm7vtL^$T(i(W%WE#yvw4-b;U8KWw$HD;25K#2{=i;zWCGRX^bjb3L17l*)XU(y6BRcKKyKz!uSc?UG%9XsL`H0rkt>Mbg_*11o%$;Mn(Db>f(1rfP4<S0=^(vO&pFyLRALn7AdXlWG$RmA{k$srF5mm#<a*^>>V4Iqzt+MZQAKjt>;SbHCXd@*cwU3+lUE&P;m>ajRV7>TL3HEa#p0*iq{n)g~J5|?@)ePkfY5tcToLxYqMQsaPOVoP4193h4yb(;kC_xV^(n^+pO7&peoo@Re(GN^u13HKIR7K9)i&iOg|FjVWhi85lC&(+m3gL_}pha$(E7J=)j>o5o+t<=^zRRGtMGfXbEg*zv6;A&5oye)13>8?;IhUZLmArjt&_^-eHonZX3#@<7r^tb1?L)BK-st+2=i^jWSMZw^NC)04s7Qo1F_-XtrJbR<;(9Jb~gW?yG(b9a%T7pP-$d~+85hBxnSv8F;HB0jli0YBV!C`b0Sz+OCp%}ZeVc`VWaUf?^>8GNJ;Jpc??p1vV@bD@zX1beP1U86zXs^KX1F5SnYP?MTEW3c|Y8#bBFm{2{d75b@|CXPNqX@#ENP->LI}y3&;Qjefy|vRyK+VG%Vm!yU2)^f$3huF>a%b%fcE@y--vDDX@)#B?3{er`uxE$h-Xbbl%F8XC$iGRLgyq6kSl1#z$Y0es^9PK5#WY!t?~NtRz%D?T!YiEtdWRfS5mB~-3}9W~s9-rhqN(LfsXsutsO}BGYGDao8ff7pA<M`x+gWfO03aSx=pN8qM9T!UC4Q${#K#rZDY<G{@j?Pr?a`V5X?TkyJvvp_4w4gVe$o-W%7yHG+J@*3O&RP<W&huc{ag29??H%=TrRPQR^Mq_I%ZXSto>sINtyZ8X9KEFZG45F+UPJ+eLq-WHgwZRkKb<M-=FVprswu=3WPi{>Hm=pS+U6ypYa-j_8x}WjcdGU(L>ZN?ly#N;l9SrmcXCAS{bWNbu6i1hUBO3$FniVs~+x0&jQ{Q!3xZgtUcL%bj<QW7f>@(_AJYi%mWXPNmJ*SCs3PGx{VSVJV&4UfBRz}VF8B3Nf5Zdg&_IX;uucyM7_yvZrdS-_By|?J6dtIHKxtv$h%tKu>xsUzAW@$uH|iLR5Du|0k}fJ3(H`M@q#g`zmWikZVmBZX4&R~p-a#&2>RX$RHc4{(}z0W4(nYX)<o>f|I&!nue!Fs{A2lO3^1_zs*!#!8pW}&h?woeEhgAJ3ASW6n4u{S<ubm%QdneUXvF2s2js==Y<w}fo7_wfRqYmLi?6NNB)G3QJxhKOsigBrx=!ToAlQB;dxrcy!+pm~i$&vt%)V&rQp3FRP`t6c*A9-))#&Q8Fv60&_1hJ%zjA-q;;nlOPTShkdk^%n61$o0W9nA3-iu{a?Zz^ucB4}oro6WqD|cMB&8s5uRthB9SUAarFOtlWuW2gzDCORLPpK)Z5pkll4=Sd)Z!8bx8!^L!e*;EnY)y4d000"
)


def _initial_schema_ddl() -> tuple[str, ...]:
    """Return the frozen 2026-08-06 DDL without importing application code."""
    snapshot = gzip.decompress(
        base64.b85decode(_INITIAL_SCHEMA_DDL_GZIP_B85)
    ).decode("utf-8")
    return tuple(statement for statement in snapshot.split(_DDL_BOUNDARY) if statement.strip())


def upgrade() -> None:
    for ddl in _initial_schema_ddl():
        op.execute(ddl)


def downgrade() -> None:
    # Reverse dependency order. Later revisions are downgraded first.
    for table in (
        "alerts", "prop_odds", "dead_letter", "staging_nflreadpy",
        "pbp_matchups", "pbp_features", "injury_history", "feature_matrix",
        "projections", "combine", "participation_player_game", "ftn_player_game",
        "ftn_play", "team_game_stats", "nextgen_stats", "depth_charts",
        "game_logs", "games", "players", "teams",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
