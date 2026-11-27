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
