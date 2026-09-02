import { createDatabase, databasePath, repositoryRoot } from "./database.ts";

const root = repositoryRoot();
const path = databasePath(root);
const db = createDatabase(root, path);
try {
  console.log(`Transit Lab control-store schema pushed at ${path}`);
} finally {
  db.close();
}
