$( af15_sound_union; Boolean-to-proposition translation. $)
${
  ml-lt-af15-union.h1 $e |- ( ph -> ch ) $.
  ml-lt-af15-union.h2 $e |- ( ps -> ch ) $.
  ml-lt-af15-union.h3 $e |- ( ph \/ ps ) $.
  ml-lt-af15-union $p |- ch $=
    wph wps wo wch ml-lt-af15-union.h3 wph wch wps ml-lt-af15-union.h1 ml-lt-af15-union.h2 jaoi ax-mp
  $.
$}

$( af15_fact_persists; Boolean-to-proposition translation. $)
${
  ml-lt-af15-persists.h1 $e |- ph $.
  ml-lt-af15-persists $p |- ( ph \/ ps ) $=
    wph wps ml-lt-af15-persists.h1 orci
  $.
$}

$( af15_no_unsupported_fire; Boolean-to-proposition translation. $)
${
  ml-lt-af15-guard.h1 $e |- ( ph -> ch ) $.
  ml-lt-af15-guard.h2 $e |- ( ps -> ch ) $.
  ml-lt-af15-guard $p |- ( ( ph \/ ps ) -> ch ) $=
    wph wch wps ml-lt-af15-guard.h1 ml-lt-af15-guard.h2 jaoi
  $.
$}

$( af17_exact_successor_cost; natural arithmetic translation. $)
${
  ml-lt-successor-cost.h1 $e |- A e. NN0 $.
  ml-lt-successor-cost $p |- ( 1 + A ) = ( A + 1 ) $=
    c1 cA ax-1cn cA cr wcel cA cc wcel cA ml-lt-successor-cost.h1 nn0rei cA recn ax-mp addcomi
  $.
$}

$( af17_robust_bellman_gap; integer arithmetic translation. $)
${
  ml-lt-bellman-gap.h1 $e |- A e. ZZ $.
  ml-lt-bellman-gap.h2 $e |- B e. ZZ $.
  ml-lt-bellman-gap.h3 $e |- C e. ZZ $.
  ml-lt-bellman-gap.h4 $e |- D e. ZZ $.
  ml-lt-bellman-gap.h5 $e |- E e. ZZ $.
  ml-lt-bellman-gap.h6 $e |- A <_ ( C + E ) $.
  ml-lt-bellman-gap.h7 $e |- ( D - E ) <_ B $.
  ml-lt-bellman-gap.h8 $e |- ( C + ( 2 x. E ) ) < D $.
  ml-lt-bellman-gap $p |- A < B $=
    cA cD cE cmin co clt wbr cD cE cmin co cB cle wbr cA cB clt wbr cA cC cE caddc co cle wbr cC cE caddc co
    cD cE cmin co clt wbr cA cD cE cmin co clt wbr ml-lt-bellman-gap.h6 cC cE caddc co cE caddc co cD clt wbr
    cC cE caddc co cD cE cmin co clt wbr cC cE caddc co cE caddc co cC cE cE caddc co caddc co cD clt cC cE
    cE cC cr wcel cC cc wcel cC ml-lt-bellman-gap.h3 zrei cC recn ax-mp cE cr wcel cE cc wcel cE
    ml-lt-bellman-gap.h5 zrei cE recn ax-mp cE cr wcel cE cc wcel cE ml-lt-bellman-gap.h5 zrei cE recn ax-mp
    addassi cC c2 cE cmul co caddc co cC cE cE caddc co caddc co cD clt c2 cE cmul co cE cE caddc co cC caddc
    cE cE cr wcel cE cc wcel cE ml-lt-bellman-gap.h5 zrei cE recn ax-mp 2timesi oveq2i ml-lt-bellman-gap.h8
    eqbrtrri eqbrtri cC cE caddc co cE cD cC cE cC ml-lt-bellman-gap.h3 zrei cE ml-lt-bellman-gap.h5 zrei
    readdcli cE ml-lt-bellman-gap.h5 zrei cD ml-lt-bellman-gap.h4 zrei ltaddsubi mpbi cA cC cE caddc co cD cE
    cmin co cA ml-lt-bellman-gap.h1 zrei cC cE cC ml-lt-bellman-gap.h3 zrei cE ml-lt-bellman-gap.h5 zrei
    readdcli cD cE cD ml-lt-bellman-gap.h4 zrei cE ml-lt-bellman-gap.h5 zrei resubcli lelttri mp2an
    ml-lt-bellman-gap.h7 cA cD cE cmin co cB cA ml-lt-bellman-gap.h1 zrei cD cE cD ml-lt-bellman-gap.h4 zrei
    cE ml-lt-bellman-gap.h5 zrei resubcli cB ml-lt-bellman-gap.h2 zrei ltletri mp2an
  $.
$}

$( af14_bounded_chain; trajectory-graph induction translation. $)
${
  $d B x $.
  $d B y $.
  $d N x $.
  $d N y $.
  $d S x $.
  $d S y $.
  $d x y $.
  ml-lt-bounded-chain-induction.h1 $e |- N e. NN0 $.
  ml-lt-bounded-chain-induction.h2 $e |- B e. NN0 $.
  ml-lt-bounded-chain-induction.h3 $e |- ( S ` 0 ) = 0 $.
  ml-lt-bounded-chain-induction.h4 $e |- A. y e. NN0 ( S ` ( y + 1 ) ) <_ ( ( S ` y ) + B ) $.
  ml-lt-bounded-chain-induction.h5 $e |- A. y e. NN0 ( ( S ` y ) e. NN0 /\ ( S ` ( y + 1 ) ) e. NN0 ) $.
  ml-lt-bounded-chain-induction $p |- ( S ` N ) <_ ( N x. B ) $=
    cN cn0 wcel cN cS cfv cN cB cmul co cle wbr ml-lt-bounded-chain-induction.h1 vx cv cS cfv vx cv cB cmul
    co cle wbr cc0 cS cfv cc0 cB cmul co cle wbr vy cv cS cfv vy cv cB cmul co cle wbr vy cv c1 caddc co cS
    cfv vy cv c1 caddc co cB cmul co cle wbr cN cS cfv cN cB cmul co cle wbr vx vy cN vx cv cc0 wceq vx cv cS
    cfv cc0 cS cfv vx cv cB cmul co cc0 cB cmul co cle vx cv cc0 wceq vx cv cc0 cS vx cv cc0 wceq id fveq2d
    vx cv cc0 wceq vx cv cc0 cB cmul vx cv cc0 wceq id oveq1d breq12d vx cv vy cv wceq vx cv cS cfv vy cv cS
    cfv vx cv cB cmul co vy cv cB cmul co cle vx cv vy cv wceq vx cv vy cv cS vx cv vy cv wceq id fveq2d vx
    cv vy cv wceq vx cv vy cv cB cmul vx cv vy cv wceq id oveq1d breq12d vx cv vy cv c1 caddc co wceq vx cv
    cS cfv vy cv c1 caddc co cS cfv vx cv cB cmul co vy cv c1 caddc co cB cmul co cle vx cv vy cv c1 caddc co
    wceq vx cv vy cv c1 caddc co cS vx cv vy cv c1 caddc co wceq id fveq2d vx cv vy cv c1 caddc co wceq vx cv
    vy cv c1 caddc co cB cmul vx cv vy cv c1 caddc co wceq id oveq1d breq12d vx cv cN wceq vx cv cS cfv cN cS
    cfv vx cv cB cmul co cN cB cmul co cle vx cv cN wceq vx cv cN cS vx cv cN wceq id fveq2d vx cv cN wceq vx
    cv cN cB cmul vx cv cN wceq id oveq1d breq12d cc0 cS cfv cc0 cc0 cB cmul co cle cc0 cS cfv cc0 cc0 cle
    ml-lt-bounded-chain-induction.h3 cc0 0re leidi eqbrtri cB cB cr wcel cB cc wcel cB
    ml-lt-bounded-chain-induction.h2 nn0rei cB recn ax-mp mul02i breqtrri vy cv cn0 wcel vy cv cS cfv vy cv
    cB cmul co cle wbr vy cv c1 caddc co cS cfv vy cv c1 caddc co cB cmul co cle wbr vy cv cn0 wcel vy cv cS
    cfv vy cv cB cmul co cle wbr wa vy cv c1 caddc co cS cfv vy cv cB cmul co cB caddc co vy cv c1 caddc co
    cB cmul co cle vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv c1 caddc co cS cfv vy cv cS
    cfv cB caddc co vy cv cB cmul co cB caddc co vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy
    cv c1 caddc co cS cfv vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cS cfv cn0 wcel vy cv
    c1 caddc co cS cfv cn0 wcel wa vy cv c1 caddc co cS cfv cn0 wcel vy cv cn0 wcel vy cv cS cfv vy cv cB
    cmul co cle wbr wa vy cv cn0 wcel vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa vy cv cn0
    wcel vy cv cS cfv vy cv cB cmul co cle wbr simpl vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel
    wa vy cn0 wral vy cv cn0 wcel vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa wi
    ml-lt-bounded-chain-induction.h5 vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa vy cn0 rsp
    ax-mp syl vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel simpr syl nn0red vy cv cn0 wcel vy cv
    cS cfv vy cv cB cmul co cle wbr wa vy cv cS cfv cB vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr
    wa vy cv cS cfv vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cS cfv cn0 wcel vy cv c1
    caddc co cS cfv cn0 wcel wa vy cv cS cfv cn0 wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa
    vy cv cn0 wcel vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa vy cv cn0 wcel vy cv cS cfv vy
    cv cB cmul co cle wbr simpl vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa vy cn0 wral vy cv
    cn0 wcel vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa wi ml-lt-bounded-chain-induction.h5
    vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa vy cn0 rsp ax-mp syl vy cv cS cfv cn0 wcel vy
    cv c1 caddc co cS cfv cn0 wcel simpl syl nn0red cB cr wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co
    cle wbr wa cB ml-lt-bounded-chain-induction.h2 nn0rei a1i readdcld vy cv cn0 wcel vy cv cS cfv vy cv cB
    cmul co cle wbr wa vy cv cB cmul co cB vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cB
    vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv vy cv cn0 wcel vy cv cS cfv vy cv cB cmul
    co cle wbr simpl nn0red cB cr wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa cB
    ml-lt-bounded-chain-induction.h2 nn0rei a1i remulcld cB cr wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul
    co cle wbr wa cB ml-lt-bounded-chain-induction.h2 nn0rei a1i readdcld vy cv cn0 wcel vy cv cS cfv vy cv
    cB cmul co cle wbr wa vy cv cn0 wcel vy cv c1 caddc co cS cfv vy cv cS cfv cB caddc co cle wbr vy cv cn0
    wcel vy cv cS cfv vy cv cB cmul co cle wbr simpl vy cv c1 caddc co cS cfv vy cv cS cfv cB caddc co cle
    wbr vy cn0 wral vy cv cn0 wcel vy cv c1 caddc co cS cfv vy cv cS cfv cB caddc co cle wbr wi
    ml-lt-bounded-chain-induction.h4 vy cv c1 caddc co cS cfv vy cv cS cfv cB caddc co cle wbr vy cn0 rsp
    ax-mp syl vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cS cfv vy cv cB cmul co cle wbr
    vy cv cS cfv cB caddc co vy cv cB cmul co cB caddc co cle wbr vy cv cn0 wcel vy cv cS cfv vy cv cB cmul
    co cle wbr simpr vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cS cfv vy cv cB cmul co cB
    vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cS cfv vy cv cn0 wcel vy cv cS cfv vy cv cB
    cmul co cle wbr wa vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel wa vy cv cS cfv cn0 wcel vy cv
    cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cn0 wcel vy cv cS cfv cn0 wcel vy cv c1 caddc co
    cS cfv cn0 wcel wa vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr simpl vy cv cS cfv cn0 wcel vy cv
    c1 caddc co cS cfv cn0 wcel wa vy cn0 wral vy cv cn0 wcel vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv
    cn0 wcel wa wi ml-lt-bounded-chain-induction.h5 vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel
    wa vy cn0 rsp ax-mp syl vy cv cS cfv cn0 wcel vy cv c1 caddc co cS cfv cn0 wcel simpl syl nn0red vy cv
    cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv cB vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co
    cle wbr wa vy cv vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr simpl nn0red cB cr wcel vy cv cn0
    wcel vy cv cS cfv vy cv cB cmul co cle wbr wa cB ml-lt-bounded-chain-induction.h2 nn0rei a1i remulcld cB
    cr wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa cB ml-lt-bounded-chain-induction.h2
    nn0rei a1i leadd1d mpbid letrd vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv c1 caddc co
    cB cmul co vy cv cB cmul co c1 cB cmul co caddc co vy cv cB cmul co cB caddc co vy cv cn0 wcel vy cv cS
    cfv vy cv cB cmul co cle wbr wa vy cv c1 cB vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv
    vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa vy cv vy cv cn0 wcel vy cv cS cfv vy cv cB cmul
    co cle wbr simpl nn0red recnd c1 cc wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa ax-1cn
    a1i cB cc wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa cB cr wcel cB cc wcel cB
    ml-lt-bounded-chain-induction.h2 nn0rei cB recn ax-mp a1i adddird vy cv cn0 wcel vy cv cS cfv vy cv cB
    cmul co cle wbr wa c1 cB cmul co cB vy cv cB cmul co caddc vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co
    cle wbr wa cB cB cc wcel vy cv cn0 wcel vy cv cS cfv vy cv cB cmul co cle wbr wa cB cr wcel cB cc wcel cB
    ml-lt-bounded-chain-induction.h2 nn0rei cB recn ax-mp a1i mullidd oveq2d eqtrd breqtrrd ex nn0ind ax-mp
  $.
$}

$( af14_horn_update_inflationary; Boolean-to-proposition translation. $)
${
  ml-lt-horn-inflationary $p |- ( ( ph \/ ( ph /\ ps ) ) <-> ph ) $=
    wph wph wps wa wo wph wph wph wph wps wa wph id wph wps simpl jaoi wph wph wps wa orc impbii
  $.
$}

$( af14_eventual_truth_persists; Boolean-to-proposition translation. $)
${
  ml-lt-eventual-truth.h1 $e |- ph $.
  ml-lt-eventual-truth $p |- ( ph \/ ps ) $=
    wph wps ml-lt-eventual-truth.h1 orci
  $.
$}
