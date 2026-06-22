// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Credential driver backed by macOS Keychain generic password items.

use std::path::PathBuf;

use openshell_core::proto::CredentialHandle;
use openshell_core::proto::credentials::v1::{
    DeleteCredentialRequest, ResolveCredentialRequest, ResolvedCredential, StoreCredentialRequest,
};
use openshell_core::{Error, Result as CoreResult};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tonic::Status;

const DEFAULT_SERVICE: &str = "com.nvidia.openshell.provider-credentials";
#[cfg(any(test, target_os = "macos"))]
const ACCOUNT_PREFIX: &str = "openshell-provider-credential";
#[cfg(any(test, target_os = "macos"))]
const HANDLE_VERSION: &str = "v1";
#[cfg(any(test, target_os = "macos"))]
const MANAGED_ACCOUNT_DIGEST_LEN: usize = 40;

pub struct MacosKeychainCredentialDriver {
    settings: MacosKeychainDriverSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MacosKeychainDriverSettings {
    service: String,
    keychain_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MacosKeychainDriverConfig {
    service: Option<String>,
    keychain_path: Option<PathBuf>,
}

impl MacosKeychainCredentialDriver {
    pub const NAME: &'static str = "macos-keychain";

    pub fn from_config(config: &toml::Table) -> CoreResult<Self> {
        let settings = MacosKeychainDriverSettings::from_table(config)?;

        #[cfg(not(target_os = "macos"))]
        {
            let _ = settings;
            return Err(Error::config(
                "macos-keychain credential driver is only supported on macOS",
            ));
        }

        #[cfg(target_os = "macos")]
        {
            Ok(Self { settings })
        }
    }

    pub async fn store_credential(
        &self,
        request: StoreCredentialRequest,
    ) -> Result<CredentialHandle, Status> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = &self.settings;
            let _ = request;
            Err(unsupported_platform_status())
        }

        #[cfg(target_os = "macos")]
        {
            let settings = self.settings.clone();
            tokio::task::spawn_blocking(move || store_credential_sync(&settings, request))
                .await
                .map_err(join_error_to_status)?
        }
    }

    pub async fn delete_credential(&self, request: DeleteCredentialRequest) -> Result<(), Status> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = &self.settings;
            let _ = request;
            Err(unsupported_platform_status())
        }

        #[cfg(target_os = "macos")]
        {
            let settings = self.settings.clone();
            tokio::task::spawn_blocking(move || delete_credential_sync(&settings, request))
                .await
                .map_err(join_error_to_status)?
        }
    }

    pub async fn resolve_credentials(
        &self,
        requests: Vec<ResolveCredentialRequest>,
    ) -> Result<Vec<ResolvedCredential>, Status> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = &self.settings;
            let _ = requests;
            Err(unsupported_platform_status())
        }

        #[cfg(target_os = "macos")]
        {
            let settings = self.settings.clone();
            tokio::task::spawn_blocking(move || resolve_credentials_sync(&settings, requests))
                .await
                .map_err(join_error_to_status)?
        }
    }

    #[cfg(target_os = "macos")]
    fn handle_from_request(
        request_id: &str,
        handle: Option<CredentialHandle>,
    ) -> Result<CredentialHandle, Status> {
        handle.ok_or_else(|| {
            Status::invalid_argument(format!(
                "macos-keychain credential request '{request_id}' is missing handle"
            ))
        })
    }

    #[cfg(any(test, target_os = "macos"))]
    fn account_from_handle(handle: &CredentialHandle) -> Result<String, Status> {
        let account = handle
            .handle
            .strip_prefix(&format!("{HANDLE_VERSION}:"))
            .ok_or_else(|| {
                Status::invalid_argument("macos-keychain credential handle is malformed")
            })?;
        validate_managed_account(account).map_err(|message| {
            Status::invalid_argument(format!(
                "macos-keychain credential handle account {message}"
            ))
        })?;
        Ok(account.to_string())
    }
}

impl std::fmt::Debug for MacosKeychainCredentialDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacosKeychainCredentialDriver")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl MacosKeychainDriverSettings {
    fn from_table(config: &toml::Table) -> CoreResult<Self> {
        let config: MacosKeychainDriverConfig = toml::Value::Table(config.clone())
            .try_into()
            .map_err(|err| {
                Error::config(format!(
                    "invalid [openshell.credential_drivers.macos-keychain]: {err}"
                ))
            })?;
        let service = config.service.as_deref().map_or_else(
            || Ok(DEFAULT_SERVICE.to_string()),
            |service| service_config("service", service),
        )?;
        if let Some(path) = config.keychain_path.as_ref() {
            if path.as_os_str().is_empty() {
                return Err(Error::config(
                    "[openshell.credential_drivers.macos-keychain] keychain_path must not be empty",
                ));
            }
            if !path.is_absolute() {
                return Err(Error::config(
                    "[openshell.credential_drivers.macos-keychain] keychain_path must be absolute",
                ));
            }
        }

        Ok(Self {
            service,
            keychain_path: config.keychain_path,
        })
    }
}

fn service_config(field_name: &str, value: &str) -> CoreResult<String> {
    let value = trimmed_config_string(field_name, value)?;
    if value.len() > 1024 {
        return Err(Error::config(format!(
            "[openshell.credential_drivers.macos-keychain] {field_name} must be 1024 bytes or fewer"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::config(format!(
            "[openshell.credential_drivers.macos-keychain] {field_name} must not contain control characters"
        )));
    }
    Ok(value.to_string())
}

fn trimmed_config_string<'a>(field_name: &str, value: &'a str) -> CoreResult<&'a str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::config(format!(
            "[openshell.credential_drivers.macos-keychain] {field_name} must not be empty"
        )));
    }
    if trimmed.len() != value.len() {
        return Err(Error::config(format!(
            "[openshell.credential_drivers.macos-keychain] {field_name} must not contain leading or trailing whitespace"
        )));
    }
    Ok(trimmed)
}

#[cfg(any(test, target_os = "macos"))]
fn managed_account(provider_name: &str, credential_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(provider_name.as_bytes());
    hasher.update([0]);
    hasher.update(credential_key.as_bytes());
    let digest = hasher.finalize();
    let hex = format!("{digest:x}");
    format!("{ACCOUNT_PREFIX}:{}", &hex[..MANAGED_ACCOUNT_DIGEST_LEN])
}

#[cfg(any(test, target_os = "macos"))]
fn validate_managed_account(account: &str) -> Result<(), &'static str> {
    let Some(digest) = account.strip_prefix(&format!("{ACCOUNT_PREFIX}:")) else {
        return Err("must use an OpenShell-managed account prefix");
    };
    if digest.len() != MANAGED_ACCOUNT_DIGEST_LEN {
        return Err("digest length is invalid");
    }
    if !digest
        .bytes()
        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err("digest must be lowercase hexadecimal");
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn unsupported_platform_status() -> Status {
    Status::unimplemented("macos-keychain credential driver is only supported on macOS")
}

#[cfg(target_os = "macos")]
fn join_error_to_status(err: tokio::task::JoinError) -> Status {
    Status::internal(format!("macOS Keychain credential task failed: {err}"))
}

#[cfg(target_os = "macos")]
fn store_credential_sync(
    settings: &MacosKeychainDriverSettings,
    request: StoreCredentialRequest,
) -> Result<CredentialHandle, Status> {
    let account = if let Some(existing_handle) = request.existing_handle.as_ref() {
        MacosKeychainCredentialDriver::account_from_handle(existing_handle)?
    } else {
        managed_account(&request.provider_name, &request.credential_key)
    };
    validate_managed_account(&account).map_err(|message| {
        Status::invalid_argument(format!(
            "macos-keychain credential handle account {message}"
        ))
    })?;
    settings.set_password(&account, request.value.as_bytes())?;
    Ok(CredentialHandle {
        driver: MacosKeychainCredentialDriver::NAME.to_string(),
        handle: format!("{HANDLE_VERSION}:{account}"),
        metadata: std::collections::HashMap::new(),
    })
}

#[cfg(target_os = "macos")]
fn delete_credential_sync(
    settings: &MacosKeychainDriverSettings,
    request: DeleteCredentialRequest,
) -> Result<(), Status> {
    let handle = MacosKeychainCredentialDriver::handle_from_request("delete", request.handle)?;
    let account = MacosKeychainCredentialDriver::account_from_handle(&handle)?;
    settings.delete_password(&account)
}

#[cfg(target_os = "macos")]
fn resolve_credentials_sync(
    settings: &MacosKeychainDriverSettings,
    requests: Vec<ResolveCredentialRequest>,
) -> Result<Vec<ResolvedCredential>, Status> {
    let mut responses = Vec::with_capacity(requests.len());
    for request in requests {
        let handle = MacosKeychainCredentialDriver::handle_from_request(
            &request.request_id,
            request.handle,
        )?;
        let account = MacosKeychainCredentialDriver::account_from_handle(&handle)?;
        let value = settings.password(&account)?;
        let value = String::from_utf8(value).map_err(|_| {
            Status::invalid_argument("macos-keychain credential value is not valid UTF-8")
        })?;
        responses.push(ResolvedCredential {
            request_id: request.request_id,
            value,
            expires_at_ms: 0,
        });
    }
    Ok(responses)
}

#[cfg(target_os = "macos")]
impl MacosKeychainDriverSettings {
    fn keychain(&self) -> Result<security_framework::os::macos::keychain::SecKeychain, Status> {
        self.keychain_path
            .as_ref()
            .map_or_else(
                security_framework::os::macos::keychain::SecKeychain::default,
                security_framework::os::macos::keychain::SecKeychain::open,
            )
            .map_err(|err| keychain_error_to_status("open keychain", err))
    }

    fn set_password(&self, account: &str, password: &[u8]) -> Result<(), Status> {
        self.keychain()?
            .set_generic_password(&self.service, account, password)
            .map_err(|err| keychain_error_to_status("store credential", err))
    }

    fn password(&self, account: &str) -> Result<Vec<u8>, Status> {
        let (password, _) = self
            .keychain()?
            .find_generic_password(&self.service, account)
            .map_err(|err| keychain_error_to_status("resolve credential", err))?;
        Ok(password.to_vec())
    }

    fn delete_password(&self, account: &str) -> Result<(), Status> {
        match self
            .keychain()?
            .find_generic_password(&self.service, account)
        {
            Ok((_, item)) => {
                item.delete();
                Ok(())
            }
            Err(err) if is_not_found(err) => Ok(()),
            Err(err) => Err(keychain_error_to_status("delete credential", err)),
        }
    }
}

#[cfg(target_os = "macos")]
fn is_not_found(err: security_framework::base::Error) -> bool {
    err.code() == ERR_SEC_ITEM_NOT_FOUND
}

#[cfg(target_os = "macos")]
fn keychain_error_to_status(operation: &str, err: security_framework::base::Error) -> Status {
    match err.code() {
        ERR_SEC_ITEM_NOT_FOUND => Status::not_found(format!(
            "macOS Keychain item was not found during {operation}"
        )),
        ERR_SEC_AUTH_FAILED => Status::unauthenticated(format!(
            "macOS Keychain authentication failed during {operation}"
        )),
        ERR_SEC_INTERACTION_NOT_ALLOWED => Status::failed_precondition(format!(
            "macOS Keychain interaction is not allowed during {operation}"
        )),
        ERR_SEC_USER_CANCELED => Status::cancelled(format!(
            "macOS Keychain prompt was canceled during {operation}"
        )),
        code => Status::unavailable(format!(
            "macOS Keychain operation failed during {operation} with OSStatus {code}: {err}"
        )),
    }
}

#[cfg(target_os = "macos")]
const ERR_SEC_USER_CANCELED: i32 = -128;
#[cfg(target_os = "macos")]
const ERR_SEC_AUTH_FAILED: i32 = -25293;
#[cfg(target_os = "macos")]
const ERR_SEC_ITEM_NOT_FOUND: i32 = -25300;
#[cfg(target_os = "macos")]
const ERR_SEC_INTERACTION_NOT_ALLOWED: i32 = -25308;

#[cfg(test)]
mod tests {
    use openshell_core::proto::CredentialHandle;
    use tonic::Code;

    use super::*;

    fn handle(value: &str) -> CredentialHandle {
        CredentialHandle {
            driver: MacosKeychainCredentialDriver::NAME.to_string(),
            handle: value.to_string(),
            metadata: std::collections::HashMap::new(),
        }
    }

    fn table(values: &[(&str, toml::Value)]) -> toml::Table {
        values
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    #[test]
    fn settings_parse_defaults() {
        let settings = MacosKeychainDriverSettings::from_table(&toml::Table::new()).unwrap();

        assert_eq!(settings.service, DEFAULT_SERVICE);
        assert!(settings.keychain_path.is_none());
    }

    #[test]
    fn settings_parse_service_and_keychain_path() {
        let settings = MacosKeychainDriverSettings::from_table(&table(&[
            (
                "service",
                toml::Value::String("com.example.openshell.test".to_string()),
            ),
            (
                "keychain_path",
                toml::Value::String("/tmp/openshell-test.keychain-db".to_string()),
            ),
        ]))
        .unwrap();

        assert_eq!(settings.service, "com.example.openshell.test");
        assert_eq!(
            settings.keychain_path.as_deref(),
            Some(std::path::Path::new("/tmp/openshell-test.keychain-db"))
        );
    }

    #[test]
    fn settings_reject_unknown_fields() {
        let err = MacosKeychainDriverSettings::from_table(&table(&[(
            "unknown",
            toml::Value::String("value".to_string()),
        )]))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn settings_reject_empty_service() {
        let err = MacosKeychainDriverSettings::from_table(&table(&[(
            "service",
            toml::Value::String(" ".to_string()),
        )]))
        .unwrap_err();

        assert!(err.to_string().contains("service"));
    }

    #[test]
    fn settings_reject_relative_keychain_path() {
        let err = MacosKeychainDriverSettings::from_table(&table(&[(
            "keychain_path",
            toml::Value::String("openshell-test.keychain-db".to_string()),
        )]))
        .unwrap_err();

        assert!(err.to_string().contains("keychain_path must be absolute"));
    }

    #[test]
    fn managed_accounts_are_stable() {
        let account = managed_account("openai-local", "OPENAI_API_KEY");

        assert!(account.starts_with(&format!("{ACCOUNT_PREFIX}:")));
        assert!(validate_managed_account(&account).is_ok());
        assert_eq!(account, managed_account("openai-local", "OPENAI_API_KEY"));
    }

    #[test]
    fn handle_resolves_account() {
        let account = managed_account("openai-local", "OPENAI_API_KEY");
        let resolved = MacosKeychainCredentialDriver::account_from_handle(&handle(&format!(
            "{HANDLE_VERSION}:{account}"
        )))
        .unwrap();

        assert_eq!(resolved, account);
    }

    #[test]
    fn handle_rejects_malformed_value() {
        let err = MacosKeychainCredentialDriver::account_from_handle(&handle(
            "openshell-provider-credential:abc",
        ))
        .unwrap_err();

        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("malformed"));
    }

    #[test]
    fn handle_rejects_unmanaged_account() {
        let err =
            MacosKeychainCredentialDriver::account_from_handle(&handle("v1:user-managed-account"))
                .unwrap_err();

        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("OpenShell-managed"));
    }

    #[test]
    fn handle_rejects_invalid_digest() {
        let err = MacosKeychainCredentialDriver::account_from_handle(&handle(&format!(
            "v1:{ACCOUNT_PREFIX}:ABCDEF0123456789012345678901234567890123"
        )))
        .unwrap_err();

        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("lowercase hexadecimal"));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[ignore = "writes to a temporary macOS Keychain; run explicitly on macOS"]
    async fn stores_resolves_overwrites_and_deletes_in_temp_keychain() {
        use security_framework::os::macos::keychain::CreateOptions;

        let dir = tempfile::tempdir().unwrap();
        let keychain_path = dir.path().join("openshell-keychain-test.keychain-db");
        let _keychain = CreateOptions::new()
            .password("openshell-test")
            .create(&keychain_path)
            .expect("create temporary keychain");
        let mut config = toml::Table::new();
        config.insert(
            "service".to_string(),
            toml::Value::String("com.nvidia.openshell.test.provider-credentials".to_string()),
        );
        config.insert(
            "keychain_path".to_string(),
            toml::Value::String(keychain_path.display().to_string()),
        );
        let driver = MacosKeychainCredentialDriver::from_config(&config).unwrap();

        let first = driver
            .store_credential(StoreCredentialRequest {
                provider_name: "openai-local".to_string(),
                credential_key: "OPENAI_API_KEY".to_string(),
                value: "sk-first".to_string(),
                existing_handle: None,
            })
            .await
            .unwrap();
        let resolved = driver
            .resolve_credentials(vec![ResolveCredentialRequest {
                request_id: "credential-0".to_string(),
                provider_name: "openai-local".to_string(),
                credential_key: "OPENAI_API_KEY".to_string(),
                handle: Some(first.clone()),
            }])
            .await
            .unwrap();
        assert_eq!(resolved[0].value, "sk-first");

        let second = driver
            .store_credential(StoreCredentialRequest {
                provider_name: "openai-local".to_string(),
                credential_key: "OPENAI_API_KEY".to_string(),
                value: "sk-second".to_string(),
                existing_handle: Some(first.clone()),
            })
            .await
            .unwrap();
        assert_eq!(first.handle, second.handle);
        let resolved = driver
            .resolve_credentials(vec![ResolveCredentialRequest {
                request_id: "credential-0".to_string(),
                provider_name: "openai-local".to_string(),
                credential_key: "OPENAI_API_KEY".to_string(),
                handle: Some(second.clone()),
            }])
            .await
            .unwrap();
        assert_eq!(resolved[0].value, "sk-second");

        driver
            .delete_credential(DeleteCredentialRequest {
                provider_name: "openai-local".to_string(),
                credential_key: "OPENAI_API_KEY".to_string(),
                handle: Some(second.clone()),
            })
            .await
            .unwrap();
        let err = driver
            .resolve_credentials(vec![ResolveCredentialRequest {
                request_id: "credential-0".to_string(),
                provider_name: "openai-local".to_string(),
                credential_key: "OPENAI_API_KEY".to_string(),
                handle: Some(second),
            }])
            .await
            .unwrap_err();
        assert_eq!(err.code(), Code::NotFound);
    }
}
