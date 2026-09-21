# VS Code 與 Codex 檔案衝突預防與修復流程

## 1. 這是什麼問題？

本專案的程式檔可能同時被兩個「寫入者」修改：

- VS Code 編輯器內尚未儲存的 buffer（分頁上有圆點）。
- Codex、ChatGPT 桌面版或其他外部程式直接寫入磁碟上的檔案。

當 VS Code 的 buffer 還沒儲存，Codex 又修改了同一個磁碟檔案，VS Code 會顯示類似下列訊息：

```text
Failed to save 'xxx.py': The content of the file is newer.
Please compare your version with the file contents or overwrite the content.
```

這不是 Git merge conflict，而是「VS Code 記憶體中的舊內容」和「磁碟上的新內容」不一致。VS Code 顯示警告是為了防止舊 buffer 意外覆蓋外部修改。

> **最重要的規則：看到衝突時先選 `Compare`，不要立即選 `Overwrite`。**

## 2. 每次請 Codex 修改程式前

使用下列交接流程，可以大幅降低衝突機率。

1. 在 VS Code 執行 **File: Save All**。
2. 確認即將請 Codex 修改的分頁沒有未儲存圆點。
3. 若你不需要繼續編輯這些檔案，關閉相關分頁。
4. 明確告訴 Codex 這次可以修改的檔案或功能範圍。
5. Codex 執行期間，不要同時修改同一批檔案。可以編輯明確不在範圍內的其他檔案。
6. Codex 完成後，先看它列出的修改檔案，再回到 VS Code。
7. 如果相關檔案原本已經開著，執行 **File: Revert File** 讓分頁重讀磁碟，或關閉分頁時選 **Don't Save** 後重新開啟。
8. 在 Source Control 視圖中檢查 diff，然後再繼續手動修改。

### 建議的合作節奏

```text
你：Save All → 停止修改目標檔案
Codex：讀取磁碟 → 修改 → 測試 → 列出修改檔案
你：Revert/Reopen 相關分頁 → 檢查 diff → 繼續編輯
```

## 3. VS Code 建議設定

保留儲存衝突保護，不要讓 VS Code 無條件覆蓋磁碟檔案。可將下列內容加入 VS Code User Settings，或在確定整個團隊都要採用後加入專案 `.vscode/settings.json`：

```jsonc
{
    // 發現磁碟檔案已被外部程式修改時，必須詢問使用者。
    "files.saveConflictResolution": "askUser",

    // 保留每次儲存的本機歷史，作為誤覆蓋時的復原點。
    "workbench.localHistory.enabled": true,
    "workbench.localHistory.maxFileEntries": 100,

    // 切換到 Codex 視窗前先儲存 buffer，降低 dirty buffer 機率。
    "files.autoSave": "onWindowChange"
}
```

設定說明：

- `files.saveConflictResolution` 必須保持 `askUser`。不要設為 `overwriteFileOnDisk`，否則會失去最後一層保護。
- `files.autoSave: onWindowChange` 適合「從 VS Code 切換到 Codex 後交付修改」的使用方式。如果你不希望自動儲存，可維持 `off`，但每次交付前一定要手動 **Save All**。
- 保留 Hot Exit 可以在 VS Code 異常關閉後拯救未儲存內容。不要為了避免衝突而平常就關閉 Hot Exit。

## 4. 發生儲存衝突時的標準流程

### 第一步：立即停止寫入

- 不要連續按 `Ctrl+S`。
- 不要先選 `Overwrite`。
- 暫停 Codex 繼續修改這個檔案。
- 不要先重啟 VS Code；重啟可能讓 Hot Exit 恢復舊 buffer，增加判斷難度。

### 第二步：選 `Compare`

VS Code 會顯示 buffer 與磁碟內容的差異。檢查兩邊各自有哪些必須保留的修改。

如果你不確定哪邊是 buffer：

1. 分頁上有未儲存圆點的是 VS Code buffer。
2. 終端、Git diff 和 Codex 目前讀到的是磁碟版本。
3. 可執行 **File: Compare Active File with Saved** 再次對照。

### 第三步：先保全 buffer 中的手動修改

如果 buffer 中有你剛寫的重要內容：

1. 只複製你要保留的片段。
2. 貼到 VS Code 的 Untitled 檔案，或專案外的暫存檔。
3. 在暫存內容安全後，才處理原檔案。

不建議把暫存檔命名成可能被 Python import 的正式模組名稱。

### 第四步：依需要選擇修復方式

#### 情況 A：要保留 Codex／磁碟版本

這是 Codex 剛完成重構後最常見的情況。

1. 確認 buffer 特有內容已複製到安全位置。
2. 執行 **File: Revert File**，或關閉分頁並選 **Don't Save**。
3. 重新開啟檔案。
4. 用 Source Control diff 確認 Codex 修改還在。

#### 情況 B：要保留 VS Code buffer

1. 先把目前磁碟版本保存為備份，或確認 Git／Local History 中有復原點。
2. 再選 `Overwrite`。
3. 立即檢查 Git diff，確認沒有把 Codex 的其他重要修改一起蓋掉。

> 只有已經比較過兩邊內容，而且確定 buffer 必須整份取代磁碟版本時，才應該選 `Overwrite`。

#### 情況 C：兩邊都要保留

1. 將 buffer 版本複製到暫存檔。
2. 對原檔案執行 **Revert File**，以磁碟版本作為基礎。
3. 使用 **Select for Compare** / **Compare with Selected** 比較暫存檔與原檔案。
4. 只手動合併所需片段，不要整份覆蓋。
5. 執行語法檢查與相關測試。

## 5. 分頁一直顯示舊內容時

先確認不是開到其他專案副本：

1. 在檔案分頁執行 **Copy Path of Active File**。
2. 確認路徑位於：

   ```text
   C:\wsn研究\wsn_project\codes\2023以恩MWSN
   ```

3. 確認分頁沒有未儲存圆點。
4. 執行 **File: Revert File**。
5. 若仍顯示舊內容，關閉該分頁後重新開啟。
6. 在所有重要 buffer 都已儲存或備份後，執行 **Developer: Reload Window**。

另外可以用下列 PowerShell 命令檢查磁碟上的真實檔案：

```powershell
Get-Item -LiteralPath '.\Algorithm\Combine\SA_SETS.py' |
    Select-Object FullName, Length, LastWriteTime

Get-FileHash -Algorithm SHA256 -LiteralPath '.\Algorithm\Combine\SA_SETS.py'

git diff -- '.\Algorithm\Combine\SA_SETS.py'
```

VS Code 畫面與上述命令不一致時，以磁碟與 Git diff 為判斷基礎，先保全 buffer 手動修改，再讓編輯器重讀檔案。

## 6. Hot Exit 持續恢復舊 buffer 時

VS Code 預設會透過 Hot Exit 保存尚未儲存的 buffer。Windows 標準安裝的備份位置是：

```text
%APPDATA%\Code\Backups
```

只在 **Revert File、關閉分頁與 Reload Window 都無法解決** 時，才進行下列最後手段：

1. 把所有必須保留的未儲存內容複製到安全位置。
2. 用 **File: Save All** 儲存其他正常檔案。
3. 關閉所有 VS Code 視窗。
4. 用工作管理員確認沒有 `Code.exe` 進程。
5. 將 `%APPDATA%\Code\Backups` 中對應專案的 session 資料夾先**改名備份**，不要立即永久刪除。
6. 重新開啟正確的專案資料夾。
7. 確認磁碟檔案正確，並執行測試。
8. 等到確定沒有遺漏的 buffer 後，才刪除改名後的備份。

注意：

- 不要在 VS Code 還開著時處理 `Backups`，否則可能被重新建立。
- 不要整個刪除 `%APPDATA%\Code\User\History`；這是 Local History，可能是最後的復原來源。
- 不要把清除 Hot Exit 備份當成一般步驟。

## 7. 已經被舊 buffer 覆蓋時如何復原

按下列順序尋找可信任版本：

1. **VS Code Local History**
   - 開啟該檔案的 Timeline。
   - 過濾為 Local History。
   - 逐筆 Compare，不要看到最新時間就直接 Restore。
   - 也可在 Command Palette 執行 **Local History: Find Entry to Restore**。
2. **Git diff／Git commit**
   - 先用 `git diff -- <path>` 看尚未 commit 的差異。
   - 如果已有可靠 commit，使用 Timeline 比較 commit 內容。
   - 在工作區有其他未完成修改時，不要使用 `git reset --hard`。
3. **Codex 修改記錄／patch**
   - 根據 Codex 列出的修改檔案與 diff 重建。
   - 先恢復單一檔案，測試後再處理下一個。
4. **Hot Exit 備份**
   - 只作為未儲存 buffer 的最後參考。
   - 必須先比較時間、路徑與內容，不能整批覆蓋專案。

## 8. 修復後驗證清單

對 Python 檔案至少完成：

- [ ] 檔案的絕對路徑正確。
- [ ] VS Code 分頁沒有未儲存圆點。
- [ ] `git diff` 只包含預期修改。
- [ ] 沒有留下 `<<<<<<<`、`=======`、`>>>>>>>` 衝突標記。
- [ ] 相關 Python 檔案可編譯／import。
- [ ] 執行對應 smoke test 與 regression test。
- [ ] 重新開啟檔案後內容仍正確。
- [ ] 重要重構已建立 Git commit 或其他明確復原點。

檢查衝突標記的命令：

```powershell
rg -n '^(<<<<<<<|=======|>>>>>>>)' .
```

## 9. 本專案的建議規範

1. 一個檔案在同一時段只由一個寫入者負責。
2. 要求 Codex 進行跨檔案重構前，先 Save All，並停止手動編輯相關檔案。
3. Codex 應在修改前讀取目前磁碟內容，修改後執行測試並回報檔案清單。
4. 如果用戶在 Codex 執行期間手動修改了同一檔案，雙方先停止寫入並比較，不自動假設任一版本可被覆蓋。
5. 一次重構完成並測試通過後，建立小而清楚的 commit，不要累積大量無復原點的變更。

## 10. 快速判斷表

| 現象 | 第一個動作 | 不要做的事 |
|---|---|---|
| `content of the file is newer` | 選 `Compare` | 立即 `Overwrite` |
| Codex 已修改，VS Code 仍顯示舊內容 | 保全手動內容後 `Revert File` | 對舊分頁按 Save |
| 分頁每次重啟都恢復舊內容 | 關閉分頁且 Don't Save，再 Reload Window | 先刪整個 Local History |
| 兩邊都有要保留的修改 | 複製 buffer 到暫存檔後手動合併 | 整份取代其中一邊 |
| 已經被覆蓋 | 查 Local History 並 Compare | 直接 `git reset --hard` |
| 懷疑開到錯誤副本 | Copy Path of Active File | 只看分頁檔名判斷 |

## 11. 官方參考資料

- [VS Code: Basic editing — Save, Auto Save, Hot Exit, Compare files](https://code.visualstudio.com/docs/editing/codebasics)
- [VS Code: User interface — Timeline 與 Local History](https://code.visualstudio.com/docs/editing/userinterface#_local-history)
- [VS Code 1.42: Save conflict resolution](https://code.visualstudio.com/updates/v1_42#_save-conflict-resolution)
- [VS Code: User and workspace settings](https://code.visualstudio.com/docs/configure/settings)
- [VS Code: Source Control quickstart — 檢查 diff 與 commit](https://code.visualstudio.com/docs/sourcecontrol/quickstart)
- [OpenAI: Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-chatgpt)
- [OpenAI: Introducing upgrades to Codex — IDE 中預覽本機修改](https://openai.com/index/introducing-upgrades-to-codex/)

---

最後更新：2026-08-02
