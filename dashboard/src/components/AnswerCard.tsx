import type { QueryResponse } from '../api/client'
import { RUNG_DESCRIPTIONS } from '../rungs'

export function AnswerCard({ result }: { result: QueryResponse }) {
  return (
    <div className="answer-card">
      <div className={`rung-badge rung-${result.rung}`}>
        <span className="rung-name">{result.rung_name}</span>
        <span className="rung-description">{RUNG_DESCRIPTIONS[result.rung_name]}</span>
      </div>

      <p className="answer-text">{result.answer}</p>

      {result.citations.length > 0 && (
        <div className="citations">
          <h3>Citations</h3>
          <ul>
            {result.citations.map((citation, index) => (
              <li key={index}>
                <p className="assertion">&ldquo;{citation.assertion}&rdquo;</p>
                <p className="source">
                  from {citation.source_id} / {citation.document_id}
                </p>
                <blockquote>{citation.text}</blockquote>
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.unsupported_assertions.length > 0 && (
        <div className="unsupported">
          <h3>Not backed by a source</h3>
          <ul>
            {result.unsupported_assertions.map((assertion, index) => (
              <li key={index}>{assertion}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
