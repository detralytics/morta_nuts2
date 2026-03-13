
def check_eurostat(mortality=None, deaths=None, population=None, print_report=True, **kwargs):
    from eurostat_checks import check_eurostat as _check
    return _check(mortality=mortality, deaths=deaths, population=population,
                  print_report=print_report, **kwargs)
