<?php
// Secure scheduler bridge called by the Render Cron Job.
// It ONLY prepares queue entries. It never dials Exotel.
require_once __DIR__ . '/voice_config.php';
require_once __DIR__ . '/voice_common.php';

header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');

if (($_SERVER['REQUEST_METHOD'] ?? '') !== 'POST') {
    voice_json_response(['ok' => false, 'error' => 'POST required'], 405);
}

$expected = defined('VOICE_SCHEDULER_SECRET') ? (string)VOICE_SCHEDULER_SECRET : '';
$provided = (string)($_SERVER['HTTP_X_SCHEDULER_SECRET'] ?? '');
if (strlen($expected) < 40 || !hash_equals($expected, $provided)) {
    voice_json_response(['ok' => false, 'error' => 'Unauthorized'], 401);
}

$dir = voice_data_dir();
if (!is_dir($dir) || !is_writable($dir)) {
    voice_json_response(['ok' => false, 'error' => 'Scheduler data folder is not writable'], 500);
}

// Prevent two cron invocations from preparing the same due schedule concurrently.
$lock = @fopen($dir . '/scheduler_tick.lock', 'c');
if (!$lock || !@flock($lock, LOCK_EX | LOCK_NB)) {
    if (is_resource($lock)) @fclose($lock);
    voice_json_response(['ok' => true, 'busy' => true, 'prepared' => 0, 'message' => 'Another scheduler run is active'], 200);
}

try {
    $schedules = voice_read('schedules.json', []);
    $queue = voice_read('call_queue.json', []);
    $now = time();
    $prepared = 0;
    $advanced = 0;

    foreach ($schedules as &$schedule) {
        if (!is_array($schedule) || ($schedule['status'] ?? '') !== 'active') continue;
        $due = strtotime((string)($schedule['next_call_at'] ?? ''));
        if (!$due || $due > $now) continue;

        $scheduleId = (string)($schedule['id'] ?? '');
        $intervalDays = max(1, min(90, (int)($schedule['interval_days'] ?? 7)));
        $customerIds = $schedule['customer_ids'] ?? [];
        if (!is_array($customerIds)) $customerIds = [];

        foreach ($customerIds as $customerId) {
            $customerId = trim((string)$customerId);
            if ($customerId === '' || $scheduleId === '') continue;
            $jobKey = $scheduleId . ':' . $customerId . ':' . date('Y-m-d', $due);
            $exists = false;
            foreach ($queue as $existing) {
                if (is_array($existing) && ($existing['job_key'] ?? '') === $jobKey) {
                    $exists = true;
                    break;
                }
            }
            if (!$exists) {
                $queue[] = [
                    'job_key' => $jobKey,
                    'schedule_id' => $scheduleId,
                    'customer_id' => $customerId,
                    'due_at' => date(DATE_ATOM, $due),
                    'status' => 'WAITING_FOR_EXOTEL_DISPATCH',
                    'prepared_at' => date(DATE_ATOM, $now)
                ];
                $prepared++;
            }
        }

        // Advance from the due time, not from current time, preserving the cadence.
        $schedule['next_call_at'] = date(DATE_ATOM, $due + ($intervalDays * 86400));
        $schedule['updated_at'] = date(DATE_ATOM, $now);
        $advanced++;
    }
    unset($schedule);

    if (!voice_write('call_queue.json', $queue)) {
        throw new RuntimeException('Could not write call_queue.json');
    }
    if (!voice_write('schedules.json', $schedules)) {
        throw new RuntimeException('Could not update schedules.json');
    }

    voice_json_response([
        'ok' => true,
        'prepared' => $prepared,
        'schedules_advanced' => $advanced,
        'calls_placed' => 0,
        'message' => 'Queue preparation only; no Exotel calls were placed.'
    ]);
} catch (Throwable $e) {
    error_log('OrderScan scheduler tick failed: ' . $e->getMessage());
    voice_json_response(['ok' => false, 'error' => 'Scheduler processing failed; check PHP error log'], 500);
} finally {
    @flock($lock, LOCK_UN);
    @fclose($lock);
}
