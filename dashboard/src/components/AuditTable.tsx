import type { AuditEntry } from '../api/client'

export function AuditTable({ entries }: { entries: AuditEntry[] }) {
  if (entries.length === 0) {
    return <p className="empty-state">No requests yet for this tenant.</p>
  }

  return (
    <table className="audit-table">
      <thead>
        <tr>
          <th>When</th>
          <th>Asked</th>
          <th>Rung</th>
          <th>Answer</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((entry) => (
          <tr key={entry.request_id}>
            <td>{new Date(entry.created_at).toLocaleString()}</td>
            <td>{entry.request_text}</td>
            <td>{entry.rung_name ?? '—'}</td>
            <td>{entry.answer ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
