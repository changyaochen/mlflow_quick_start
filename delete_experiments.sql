-- Permanently delete the marked `deleted` experiments
DELETE
FROM experiment_tags
WHERE experiment_id IN (
  SELECT experiment_id
  FROM experiments
  WHERE lifecycle_stage = 'deleted'
);

DELETE
FROM latest_metrics
WHERE run_uuid IN (
  SELECT run_uuid
  FROM runs
  WHERE experiment_id IN (
    SELECT experiment_id
    FROM experiments
    WHERE lifecycle_stage = 'deleted'
  )
);

DELETE
FROM metrics WHERE run_uuid IN (
  SELECT run_uuid
  FROM runs WHERE experiment_id IN (
    SELECT experiment_id
    FROM experiments
    WHERE lifecycle_stage = 'deleted'
  )
);

DELETE
FROM tags
WHERE run_uuid IN (
  SELECT run_uuid
  FROM runs
  WHERE experiment_id IN (
    SELECT experiment_id
    FROM experiments
    WHERE lifecycle_stage = 'deleted'
  )
);

DELETE
FROM tags WHERE run_uuid IN (
  SELECT run_uuid
  FROM runs
  WHERE experiment_id IN (
    SELECT experiment_id
    FROM experiments
    WHERE lifecycle_stage = 'deleted'
  )
);

DELETE
FROM params
WHERE run_uuid IN (
  SELECT run_uuid
  FROM experiments
  WHERE lifecycle_stage = 'deleted'
);

DELETE
FROM runs
WHERE run_uuid IN (
  SELECT run_uuid
  FROM experiments
  WHERE lifecycle_stage = 'deleted'
);

DELETE
FROM experiments
WHERE lifecycle_stage = 'deleted';