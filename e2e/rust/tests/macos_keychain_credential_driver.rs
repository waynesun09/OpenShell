// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "e2e-macos-keychain", target_os = "macos"))]

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use openshell_e2e::harness::binary::openshell_cmd;
use openshell_e2e::harness::cli::run_cli;
use openshell_e2e::harness::output::strip_ansi;
use sha2::{Digest, Sha256};

const ACCOUNT_PREFIX: &str = "openshell-provider-credential";
const CREDENTIAL_KEY: &str = "OPENAI_API_KEY";
const DEFAULT_SERVICE: &str = "com.nvidia.openshell.e2e.provider-credentials";

fn unique_suffix() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{}-{millis}", std::process::id())
}

fn keychain_path() -> Option<PathBuf> {
    std::env::var_os("OPENSHELL_E2E_MACOS_KEYCHAIN_PATH").map(PathBuf::from)
}

fn keychain_service() -> String {
    std::env::var("OPENSHELL_E2E_MACOS_KEYCHAIN_SERVICE")
        .unwrap_or_else(|_| DEFAULT_SERVICE.to_string())
}

fn managed_keychain_account(provider_name: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(provider_name.as_bytes());
    hasher.update([0]);
    hasher.update(CREDENTIAL_KEY.as_bytes());
    let digest = hasher.finalize();
    let hex = format!("{digest:x}");
    format!("{ACCOUNT_PREFIX}:{}", &hex[..40])
}

async fn delete_provider(name: &str) {
    let mut cmd = openshell_cmd();
    cmd.arg("provider")
        .arg("delete")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let _ = cmd.status().await;
}

async fn create_provider(name: &str, secret_value: &str) -> Result<(), String> {
    let credential = format!("{CREDENTIAL_KEY}={secret_value}");
    let (output, code) = run_cli(&[
        "provider",
        "create",
        "--name",
        name,
        "--type",
        "openai",
        "--credential",
        &credential,
    ])
    .await;
    let clean = strip_ansi(&output);
    if code != 0 {
        return Err(format!("provider create {name} failed (exit {code}):\n{clean}"));
    }
    Ok(())
}

async fn update_provider(name: &str, secret_value: &str) -> Result<(), String> {
    let credential = format!("{CREDENTIAL_KEY}={secret_value}");
    let (output, code) = run_cli(&["provider", "update", name, "--credential", &credential]).await;
    let clean = strip_ansi(&output);
    if code != 0 {
        return Err(format!("provider update {name} failed (exit {code}):\n{clean}"));
    }
    Ok(())
}

async fn assert_provider_get_does_not_expose_secret(
    provider_name: &str,
    secret_value: &str,
) -> Result<(), String> {
    let (output, code) = run_cli(&["provider", "get", provider_name]).await;
    let clean = strip_ansi(&output);
    if code != 0 {
        return Err(format!(
            "provider get {provider_name} failed (exit {code}):\n{clean}"
        ));
    }
    if clean.contains(secret_value) {
        return Err(format!(
            "provider get {provider_name} exposed credential material"
        ));
    }
    Ok(())
}

async fn security_command(args: &[String]) -> Result<String, String> {
    let output = tokio::process::Command::new("security")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|err| format!("failed to spawn security {args:?}: {err}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(format!(
            "security {args:?} failed (exit {:?}):\n{stderr}",
            output.status.code()
        ));
    }
    Ok(stdout)
}

async fn keychain_password(
    service: &str,
    account: &str,
    keychain: Option<&str>,
) -> Result<String, String> {
    let mut args = vec![
        "find-generic-password".to_string(),
        "-s".to_string(),
        service.to_string(),
        "-a".to_string(),
        account.to_string(),
        "-w".to_string(),
    ];
    if let Some(keychain) = keychain {
        args.push(keychain.to_string());
    }

    security_command(&args)
    .await
    .map(|value| value.trim_end_matches(['\r', '\n']).to_string())
}

async fn delete_keychain_item(service: &str, account: &str, keychain: Option<&str>) {
    let mut args = vec![
        "delete-generic-password".to_string(),
        "-s".to_string(),
        service.to_string(),
        "-a".to_string(),
        account.to_string(),
    ];
    if let Some(keychain) = keychain {
        args.push(keychain.to_string());
    }

    let _ = security_command(&args).await;
}

async fn assert_keychain_value(
    service: &str,
    account: &str,
    keychain: Option<&str>,
    expected_value: &str,
) -> Result<(), String> {
    let value = keychain_password(service, account, keychain).await?;
    if value != expected_value {
        return Err("macOS Keychain stored an unexpected credential value".to_string());
    }
    Ok(())
}

async fn assert_keychain_item_deleted(
    service: &str,
    account: &str,
    keychain: Option<&str>,
) -> Result<(), String> {
    match keychain_password(service, account, keychain).await {
        Ok(_) => Err("macOS Keychain item still exists after provider deletion".to_string()),
        Err(_) => Ok(()),
    }
}

#[tokio::test]
async fn provider_credentials_are_stored_in_macos_keychain() {
    assert!(
        matches!(
            std::env::var("OPENSHELL_E2E_MACOS_KEYCHAIN_CREDENTIAL_DRIVER").as_deref(),
            Ok("1")
        ),
        "run with `mise run e2e:macos-keychain` so the Docker wrapper enables the macos-keychain credential storage driver"
    );

    let keychain = keychain_path();
    if let Some(keychain) = keychain.as_ref() {
        assert!(
            keychain.exists(),
            "temporary Keychain does not exist at {}",
            keychain.display()
        );
    }
    let keychain = keychain.as_ref().map(|path| path.display().to_string());
    let service = keychain_service();
    let suffix = unique_suffix();
    let provider_name = format!("macos-keychain-{suffix}");
    let account = managed_keychain_account(&provider_name);
    let secret_first = format!("sk-e2e-macos-keychain-first-{suffix}");
    let secret_second = format!("sk-e2e-macos-keychain-second-{suffix}");

    delete_provider(&provider_name).await;
    delete_keychain_item(&service, &account, keychain.as_deref()).await;

    let result: Result<(), String> = async {
        create_provider(&provider_name, &secret_first).await?;
        assert_provider_get_does_not_expose_secret(&provider_name, &secret_first).await?;
        assert_keychain_value(&service, &account, keychain.as_deref(), &secret_first).await?;

        update_provider(&provider_name, &secret_second).await?;
        assert_provider_get_does_not_expose_secret(&provider_name, &secret_second).await?;
        assert_keychain_value(&service, &account, keychain.as_deref(), &secret_second).await?;
        Ok(())
    }
    .await;

    delete_provider(&provider_name).await;
    if result.is_ok() {
        assert_keychain_item_deleted(&service, &account, keychain.as_deref())
            .await
            .expect("credential backend item should be deleted with provider");
    } else {
        delete_keychain_item(&service, &account, keychain.as_deref()).await;
    }
    result.expect("macOS Keychain credential storage e2e failed");
}
