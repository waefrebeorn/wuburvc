/*
 * test_bridge_rc — Regression harness for the cmd.exe system() bridge.
 *
 * The WuBuRVC engine calls Python's extract_rvc_weights.py via system() which
 * on Windows is cmd.exe /c.  Two bugs (2026-08-09) made this always rc=1:
 *
 *   1. The python-probe loop `for (pi=0; py_candidates[pi] && !py; pi++)`
 *      short-circuited on the first NULL entry (getenv("WUBU_PYTHON") unset)
 *      and never probed the venv paths → fell back to bare "python" (not on
 *      cmd's PATH) → rc=1.
 *
 *   2. The command string had multiple quoted segments and did not end in a
 *      quote.  cmd.exe's /c parser mangles such lines ("The filename,
 *      directory name, or volume label syntax is incorrect").  The fix is to
 *      wrap the whole command in an extra pair of double quotes.
 *
 * This test reproduces the exact command the engine builds and asserts rc==0.
 * It only runs when the WuBuMedia venv python exists; otherwise it is a
 * no-op PASS (skipped).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

int main(void) {
    const char *py      = "C:\\Users\\eman5\\WuBuMedia\\.venv_win\\Scripts\\python.exe";
    const char *script  = "C:\\Users\\eman5\\wuburvc\\tools\\extract_rvc_weights.py";
    const char *model   = "C:\\Users\\eman5\\WuBuMedia\\models\\rvc\\mj83k\\model.pth";
    const char *bin     = "C:\\Users\\eman5\\WuBuMedia\\models\\rvc\\mj83k\\model.pth.weights.bin";

    /* Skip if prerequisites aren't present (e.g. CI without the model). */
    if (!file_exists(py) || !file_exists(script) || !file_exists(model)) {
        printf("TEST_SKIP: wubu_bridge_rc — prerequisites missing\n");
        return 0;
    }

    /* Pre-remove the .bin so we test extraction, not just existence. */
    remove(bin);

    char cmd[2048];
    /* NEW (fixed): whole command wrapped in extra quotes — cmd strips
     * first+last and re-parses correctly. */
    snprintf(cmd, sizeof(cmd),
             "\"\"%s\" \"%s\" \"%s\" \"%s\" 2>nul\"",
             py, script, model, bin);

    int rc = system(cmd);
    int bin_created = file_exists(bin) && (rc == 0);

    printf("TEST_%s: wubu_bridge_rc — system() rc=%d, bin_created=%d\n",
           bin_created ? "PASS" : "FAIL", rc, bin_created);
    return bin_created ? 0 : 1;
}
