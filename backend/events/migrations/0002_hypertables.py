from django.db import migrations

SQL = """
SELECT create_hypertable('events_posturerow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_presencerow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_vitalrow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
SELECT create_hypertable('events_occupancyrow','ts',chunk_time_interval=>86400000,if_not_exists=>TRUE,migrate_data=>TRUE);
"""

def apply_hypertables(apps, schema_editor):
    if schema_editor.connection.vendor != 'postgresql':
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname='timescaledb'")
        if cursor.fetchone() is None:
            return  # TimescaleDB 未安装，静默跳过
        cursor.execute(SQL)

class Migration(migrations.Migration):
    dependencies = [("events","0001_initial")]
    operations = [migrations.RunPython(apply_hypertables, reverse_code=migrations.RunPython.noop)]
