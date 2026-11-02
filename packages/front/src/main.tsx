import { render } from 'preact';
import { App } from './ui/App';
import './ui/agent-api';
import './ui/style.css';

render(<App />, document.getElementById('app')!);
