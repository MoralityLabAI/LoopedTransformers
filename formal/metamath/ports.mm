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
