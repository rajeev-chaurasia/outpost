import { useState, type FormEvent } from 'react'

interface Props {
  onSubmit: (question: string) => void
  isLoading: boolean
}

export function QueryPanel({ onSubmit, isLoading }: Props) {
  const [question, setQuestion] = useState('')

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const trimmed = question.trim()
    if (trimmed) onSubmit(trimmed)
  }

  return (
    <form className="query-panel" onSubmit={handleSubmit}>
      <textarea
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        placeholder="Ask a question about this tenant's data..."
        rows={3}
      />
      <button type="submit" disabled={isLoading || !question.trim()}>
        {isLoading ? 'Asking…' : 'Ask'}
      </button>
    </form>
  )
}
