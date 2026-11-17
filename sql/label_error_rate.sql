SELECT
    gold_label,
    COUNT(*) AS sample_count,
    SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) AS badcase_count,
    ROUND(
        1.0 * SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) / COUNT(*),
        4
    ) AS error_rate
FROM predictions
GROUP BY gold_label
ORDER BY error_rate DESC, gold_label;
