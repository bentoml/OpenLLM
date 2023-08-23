from __future__ import annotations
import os

from hypothesis import HealthCheck, settings
settings.register_profile('CI', settings(suppress_health_check=[HealthCheck.too_slow]), deadline=None)

if 'CI' in os.environ: settings.load_profile('CI')
